#!/usr/bin/env crystal

require "db"
require "option_parser"
require "sqlite3"

module IMMTree
  SCHEMA_VERSION = "2"
  FORWARD = 0
  REVERSE = 1
  KEY_SCHEMA = "key"
  ID_SCHEMA = "id"

  class BuildError < Exception
  end

  class Builder
    getter input_path : String
    getter output_path : String
    getter max_depth : Int32
    getter batch_size : Int32
    getter force : Bool
    getter directions : Array(Int32)
    getter schema : String

    def initialize(@input_path, @output_path, @max_depth, @batch_size, @force, @directions, @schema)
      raise BuildError.new("--depth must be positive") unless @max_depth > 0
      raise BuildError.new("--batch-size must be positive") unless @batch_size > 0
      raise BuildError.new("--direction selected no trees") if @directions.empty?
      raise BuildError.new("--schema must be key or id") unless {KEY_SCHEMA, ID_SCHEMA}.includes?(@schema)
      raise BuildError.new("input does not exist: #{@input_path}") unless File.file?(@input_path)
    end

    def run : Nil
      output = File.expand_path(@output_path)
      if File.exists?(output)
        if @force
          File.delete(output)
        else
          raise BuildError.new("output already exists: #{output} (use --force to replace it)")
        end
      end

      corpus = File.read(@input_path).to_slice
      if corpus.size <= @max_depth
        raise BuildError.new("input must be larger than --depth when tail positions are omitted")
      end
      observations_by_direction = Hash(Int32, Int64).new(0_i64)

      DB.open("sqlite3://#{output}") do |db|
        configure(db)
        create_schema(db)
        @directions.each do |direction|
          source = direction == REVERSE ? reversed(corpus) : corpus
          observations_by_direction[direction] = load_raw_nodes(db, source, direction)
        end
        compact_nodes(db)
        write_meta(db, corpus.size, observations_by_direction)
        db.exec "ANALYZE"
      end

      observations = observations_by_direction.values.sum
      STDERR.puts "built #{@output_path}: #{corpus.size} bytes, #{observations} observations, depth #{@max_depth}"
    end

    private def configure(db : DB::Database) : Nil
      db.exec "PRAGMA journal_mode = OFF"
      db.exec "PRAGMA synchronous = OFF"
      db.exec "PRAGMA temp_store = FILE"
      db.exec "PRAGMA locking_mode = EXCLUSIVE"
      db.exec "PRAGMA cache_size = -200000"
    end

    private def create_schema(db : DB::Database) : Nil
      if @schema == KEY_SCHEMA
        create_key_schema(db)
      else
        create_id_schema(db)
      end

      db.exec <<-SQL
        CREATE TABLE meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID
      SQL

      db.exec <<-SQL
        CREATE TEMP TABLE raw_nodes (
          direction INTEGER NOT NULL,
          parent_key BLOB NOT NULL,
          token INTEGER NOT NULL
        )
      SQL
    end

    private def create_key_schema(db : DB::Database) : Nil
      db.exec <<-SQL
        CREATE TABLE nodes (
          direction INTEGER NOT NULL CHECK (direction IN (0, 1)),
          parent_key BLOB NOT NULL,
          token INTEGER NOT NULL CHECK (token BETWEEN 0 AND 255),
          mass INTEGER NOT NULL CHECK (mass > 0),
          PRIMARY KEY (direction, parent_key, token)
        ) WITHOUT ROWID
      SQL
    end

    private def create_id_schema(db : DB::Database) : Nil
      db.exec <<-SQL
        CREATE TABLE nodes (
          id INTEGER NOT NULL PRIMARY KEY,
          direction INTEGER NOT NULL CHECK (direction IN (0, 1)),
          parent_id INTEGER NOT NULL,
          token INTEGER NOT NULL CHECK (token BETWEEN 0 AND 255),
          depth INTEGER NOT NULL CHECK (depth > 0),
          key BLOB NOT NULL,
          mass INTEGER NOT NULL CHECK (mass > 0),
          UNIQUE(direction, key),
          UNIQUE(direction, parent_id, token)
        ) WITHOUT ROWID
      SQL
    end

    private def load_raw_nodes(db : DB::Database, corpus : Bytes, direction : Int32) : Int64
      observations = 0_i64
      start_limit = corpus.size - @max_depth
      label = direction == FORWARD ? "forward" : "reverse"

      db.exec "BEGIN"
      begin
        i = 0
        while i < start_limit
          depth = 1
          while depth <= @max_depth
            parent_len = depth - 1
            token = corpus[i + parent_len].to_i
            if parent_len == 0
              db.exec "INSERT INTO raw_nodes(direction, parent_key, token) VALUES (?, X'', ?)", direction, token
            else
              parent_key = corpus[i, parent_len]
              db.exec "INSERT INTO raw_nodes(direction, parent_key, token) VALUES (?, ?, ?)", direction, parent_key, token
            end

            observations += 1
            if observations % @batch_size == 0
              db.exec "COMMIT"
              STDERR.puts "loaded #{label} #{observations} observations"
              db.exec "BEGIN"
            end

            depth += 1
          end
          i += 1
        end
        db.exec "COMMIT"
      rescue ex
        db.exec "ROLLBACK" rescue nil
        raise ex
      end

      observations
    end

    private def compact_nodes(db : DB::Database) : Nil
      STDERR.puts "compacting raw observations"
      if @schema == KEY_SCHEMA
        compact_key_nodes(db)
      else
        compact_id_nodes(db)
      end
    end

    private def compact_key_nodes(db : DB::Database) : Nil
      db.exec <<-SQL
        INSERT INTO nodes(direction, parent_key, token, mass)
        SELECT direction, parent_key, token, COUNT(*) AS mass
        FROM raw_nodes
        GROUP BY direction, parent_key, token
      SQL
    end

    private def compact_id_nodes(db : DB::Database) : Nil
      db.exec <<-SQL
        CREATE TEMP TABLE compact_nodes AS
        WITH grouped AS (
          SELECT
            direction,
            parent_key,
            token,
            unhex(hex(parent_key) || printf('%02X', token)) AS key,
            length(parent_key) + 1 AS depth,
            COUNT(*) AS mass
          FROM raw_nodes
          GROUP BY direction, parent_key, token
        )
        SELECT
          ROW_NUMBER() OVER (ORDER BY direction, depth, key) AS id,
          direction,
          parent_key,
          token,
          key,
          depth,
          mass
        FROM grouped
      SQL

      db.exec "CREATE INDEX compact_nodes_key_idx ON compact_nodes(direction, key)"

      db.exec <<-SQL
        INSERT INTO nodes(id, direction, parent_id, token, depth, key, mass)
        SELECT
          c.id,
          c.direction,
          COALESCE(p.id, 0) AS parent_id,
          c.token,
          c.depth,
          c.key,
          c.mass
        FROM compact_nodes c
        LEFT JOIN compact_nodes p
          ON p.direction = c.direction
         AND p.key = c.parent_key
        ORDER BY c.id
      SQL
    end

    private def write_meta(db : DB::Database, corpus_bytes : Int32, observations_by_direction : Hash(Int32, Int64)) : Nil
      observations = observations_by_direction.values.sum
      direction_names = observations_by_direction.keys.sort.map { |direction| direction == FORWARD ? "forward" : "reverse" }
      entries = {
        "schema_version" => SCHEMA_VERSION,
        "schema" => @schema,
        "tokenization" => "raw-bytes",
        "source_path" => File.expand_path(@input_path),
        "source_bytes" => corpus_bytes.to_s,
        "max_depth" => @max_depth.to_s,
        "recorded_starts" => (corpus_bytes - @max_depth).to_s,
        "directions" => direction_names.join(","),
        "forward_observations" => observations_by_direction[FORWARD].to_s,
        "reverse_observations" => observations_by_direction[REVERSE].to_s,
        "observations" => observations.to_s,
      }

      entries.each do |key, value|
        db.exec "INSERT INTO meta(key, value) VALUES (?, ?)", key, value
      end
    end

    private def reversed(corpus : Bytes) : Bytes
      out = Bytes.new(corpus.size)
      last = corpus.size - 1
      i = 0
      while i < corpus.size
        out[i] = corpus[last - i]
        i += 1
      end
      out
    end
  end
end

input_path = nil
output_path = nil
max_depth = 8
batch_size = 250_000
force = false
direction_arg = "both"
schema = IMMTree::KEY_SCHEMA

parser = OptionParser.new do |p|
  p.banner = "Usage: imm_tree_build --input PATH --output PATH --depth N [--force]"

  p.on("-i PATH", "--input=PATH", "Corpus text file to index") { |path| input_path = path }
  p.on("-o PATH", "--output=PATH", "SQLite database to create") { |path| output_path = path }
  p.on("-d N", "--depth=N", "Maximum context depth") { |n| max_depth = n.to_i }
  p.on("--batch-size=N", "Inserted observations per transaction") { |n| batch_size = n.to_i }
  p.on("--direction=NAME", "forward, reverse, or both (default: both)") { |name| direction_arg = name }
  p.on("--schema=NAME", "key or id (default: key)") { |name| schema = name }
  p.on("--force", "Replace an existing output database") { force = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

begin
  parser.parse
  raise IMMTree::BuildError.new("--input is required") unless input_path
  raise IMMTree::BuildError.new("--output is required") unless output_path
  directions = case direction_arg
               when "forward"
                 [IMMTree::FORWARD]
               when "reverse"
                 [IMMTree::REVERSE]
               when "both"
                 [IMMTree::FORWARD, IMMTree::REVERSE]
               else
                 raise IMMTree::BuildError.new("--direction must be forward, reverse, or both")
               end

  IMMTree::Builder.new(
    input_path.not_nil!,
    output_path.not_nil!,
    max_depth,
    batch_size,
    force,
    directions,
    schema
  ).run
rescue ex : IMMTree::BuildError
  STDERR.puts "error: #{ex.message}"
  STDERR.puts parser
  exit 2
end
