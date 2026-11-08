#!/usr/bin/env crystal

require "db"
require "option_parser"
require "sqlite3"

module IMMHorizon
  FORWARD = 0
  REVERSE = 1

  class BuildError < Exception
  end

  class Builder
    getter db_path : String
    getter input_path : String
    getter horizon : Int32
    getter flush_entries : Int32
    getter replace : Bool
    getter directions : Array(Int32)

    def initialize(@db_path, @input_path, @horizon, @flush_entries, @replace, @directions)
      raise BuildError.new("--horizon must be positive") unless @horizon > 0
      raise BuildError.new("--flush-entries must be positive") unless @flush_entries > 0
      raise BuildError.new("--direction selected no trees") if @directions.empty?
      raise BuildError.new("database does not exist: #{@db_path}") unless File.file?(@db_path)
      raise BuildError.new("input does not exist: #{@input_path}") unless File.file?(@input_path)
    end

    def run : Nil
      corpus = File.read(@input_path).to_slice

      DB.open("sqlite3://#{File.expand_path(@db_path)}") do |db|
        configure(db)
        assert_id_schema(db)
        max_depth = meta_int(db, "max_depth")
        if corpus.size < max_depth + @horizon
          raise BuildError.new("input must be at least depth + horizon bytes")
        end

        create_schema(db)
        prepare_existing_counts(db)

        @directions.each do |direction|
          source = direction == REVERSE ? reversed(corpus) : corpus
          node_ids = load_node_ids(db, direction)
          build_direction(db, source, direction, max_depth, node_ids)
        end

        write_meta(db, corpus.size, max_depth)
        db.exec "ANALYZE horizon_counts"
      end
    end

    private def configure(db : DB::Database) : Nil
      db.exec "PRAGMA journal_mode = WAL"
      db.exec "PRAGMA synchronous = NORMAL"
      db.exec "PRAGMA temp_store = FILE"
      db.exec "PRAGMA cache_size = -200000"
    end

    private def assert_id_schema(db : DB::Database) : Nil
      schema = meta_string(db, "schema")
      raise BuildError.new("horizon counts require a DB built with --schema id") unless schema == "id"
    end

    private def create_schema(db : DB::Database) : Nil
      db.exec <<-SQL
        CREATE TABLE IF NOT EXISTS horizon_counts (
          node_id INTEGER NOT NULL,
          horizon INTEGER NOT NULL CHECK (horizon > 0),
          target INTEGER NOT NULL CHECK (target BETWEEN 0 AND 255),
          mass INTEGER NOT NULL CHECK (mass > 0),
          PRIMARY KEY (node_id, horizon, target)
        ) WITHOUT ROWID
      SQL

      db.exec "CREATE INDEX IF NOT EXISTS horizon_counts_horizon_idx ON horizon_counts(horizon, target)"
    end

    private def prepare_existing_counts(db : DB::Database) : Nil
      existing = db.query_one("SELECT COUNT(*) FROM horizon_counts", as: Int64)
      return if existing == 0

      if @replace
        db.exec "DELETE FROM horizon_counts"
      else
        raise BuildError.new("horizon_counts already has #{existing} rows (use --replace)")
      end
    end

    private def load_node_ids(db : DB::Database, direction : Int32) : Hash(String, Int64)
      label = direction == FORWARD ? "forward" : "reverse"
      STDERR.puts "loading #{label} node ids"

      ids = Hash(String, Int64).new
      db.query "SELECT id, hex(key) FROM nodes WHERE direction = ?", direction do |rs|
        rs.each do
          id = rs.read(Int64)
          key_hex = rs.read(String)
          ids[key_hex] = id
        end
      end
      ids
    end

    private def build_direction(db : DB::Database, corpus : Bytes, direction : Int32, max_depth : Int32, node_ids : Hash(String, Int64)) : Nil
      label = direction == FORWARD ? "forward" : "reverse"
      start_limit = corpus.size - max_depth - @horizon + 1
      hex = hex_table
      counts = Hash(UInt64, Int32).new(0)
      observations = 0_i64

      i = 0
      while i < start_limit
        key_hex = String.new
        depth = 1
        while depth <= max_depth
          token = corpus[i + depth - 1]
          key_hex += hex[token]
          node_id = node_ids[key_hex]?
          raise BuildError.new("missing node id for #{label} key #{key_hex}") unless node_id

          h = 1
          while h <= @horizon
            target = corpus[i + depth - 1 + h]
            packed = pack(node_id, h, target)
            counts[packed] += 1
            observations += 1

            if counts.size >= @flush_entries
              flush_counts(db, counts)
              STDERR.puts "flushed #{label} #{observations} horizon observations"
            end

            h += 1
          end

          depth += 1
        end
        i += 1
      end

      flush_counts(db, counts)
      STDERR.puts "built #{label} #{observations} horizon observations"
    end

    private def flush_counts(db : DB::Database, counts : Hash(UInt64, Int32)) : Nil
      return if counts.empty?

      db.exec "BEGIN"
      begin
        counts.each do |packed, mass|
          node_id, horizon, target = unpack(packed)
          db.exec <<-SQL, node_id, horizon, target, mass
            INSERT INTO horizon_counts(node_id, horizon, target, mass)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id, horizon, target)
            DO UPDATE SET mass = mass + excluded.mass
          SQL
        end
        db.exec "COMMIT"
      rescue ex
        db.exec "ROLLBACK" rescue nil
        raise ex
      ensure
        counts.clear
      end
    end

    private def write_meta(db : DB::Database, corpus_bytes : Int32, max_depth : Int32) : Nil
      horizon_starts = corpus_bytes - max_depth - @horizon + 1
      entries = {
        "horizon_max" => @horizon.to_s,
        "horizon_recorded_starts" => horizon_starts.to_s,
        "horizon_directions" => @directions.sort.map { |d| d == FORWARD ? "forward" : "reverse" }.join(","),
      }

      entries.each do |key, value|
        db.exec <<-SQL, key, value
          INSERT INTO meta(key, value) VALUES (?, ?)
          ON CONFLICT(key) DO UPDATE SET value = excluded.value
        SQL
      end
    end

    private def meta_string(db : DB::Database, key : String) : String
      value = db.query_one?("SELECT value FROM meta WHERE key = ?", key, as: String)
      raise BuildError.new("missing meta key: #{key}") unless value
      value
    end

    private def meta_int(db : DB::Database, key : String) : Int32
      meta_string(db, key).to_i
    end

    private def pack(node_id : Int64, horizon : Int32, target : UInt8) : UInt64
      ((node_id.to_u64) << 16) | ((horizon.to_u64) << 8) | target.to_u64
    end

    private def unpack(value : UInt64) : Tuple(Int64, Int32, Int32)
      node_id = (value >> 16).to_i64
      horizon = ((value >> 8) & 0xff_u64).to_i
      target = (value & 0xff_u64).to_i
      {node_id, horizon, target}
    end

    private def hex_table : Array(String)
      (0..255).map { |byte| byte.to_s(16).rjust(2, '0').upcase }
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

db_path = nil
input_path = nil
horizon = 4
flush_entries = 250_000
replace = false
direction_arg = "forward"

parser = OptionParser.new do |p|
  p.banner = "Usage: imm_horizon_build --db PATH --input PATH --horizon N [--replace]"

  p.on("--db=PATH", "ID-schema SQLite tree database to augment") { |path| db_path = path }
  p.on("-i PATH", "--input=PATH", "Corpus text file used to build the tree") { |path| input_path = path }
  p.on("--horizon=N", "Maximum future offset to count") { |n| horizon = n.to_i }
  p.on("--flush-entries=N", "Unique count keys to hold before flushing") { |n| flush_entries = n.to_i }
  p.on("--direction=NAME", "forward, reverse, or both (default: forward)") { |name| direction_arg = name }
  p.on("--replace", "Replace existing horizon_counts rows") { replace = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

begin
  parser.parse
  raise IMMHorizon::BuildError.new("--db is required") unless db_path
  raise IMMHorizon::BuildError.new("--input is required") unless input_path

  directions = case direction_arg
               when "forward"
                 [IMMHorizon::FORWARD]
               when "reverse"
                 [IMMHorizon::REVERSE]
               when "both"
                 [IMMHorizon::FORWARD, IMMHorizon::REVERSE]
               else
                 raise IMMHorizon::BuildError.new("--direction must be forward, reverse, or both")
               end

  IMMHorizon::Builder.new(
    db_path.not_nil!,
    input_path.not_nil!,
    horizon,
    flush_entries,
    replace,
    directions
  ).run
rescue ex : IMMHorizon::BuildError
  STDERR.puts "error: #{ex.message}"
  STDERR.puts parser
  exit 2
end
