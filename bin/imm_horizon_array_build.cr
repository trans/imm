#!/usr/bin/env crystal

require "db"
require "file_utils"
require "option_parser"
require "sqlite3"

module IMMHorizonArray
  FORWARD = 0
  REVERSE = 1

  class BuildError < Exception
  end

  record NodeSet, ids : Array(Int64), row_by_id : Hash(Int64, Int32), id_by_key : Hash(String, Int64)

  class Builder
    getter db_path : String
    getter input_path : String
    getter horizon : Int32
    getter replace : Bool
    getter directions : Array(Int32)
    getter tmp_dir : String

    @total_path : String
    @entropy_path : String
    @max_target_path : String
    @max_mass_path : String
    @target_count_path : String

    def initialize(@db_path, @input_path, @horizon, @replace, @directions, @tmp_dir)
      raise BuildError.new("--horizon must be positive") unless @horizon > 0
      raise BuildError.new("--horizon must fit in one byte for the current packed counter") unless @horizon <= 255
      raise BuildError.new("--direction selected no trees") if @directions.empty?
      raise BuildError.new("database does not exist: #{@db_path}") unless File.file?(@db_path)
      raise BuildError.new("input does not exist: #{@input_path}") unless File.file?(@input_path)

      stamp = "#{Process.pid}-#{Time.utc.to_unix_ms}"
      @total_path = File.join(@tmp_dir, "imm-horizon-total-#{stamp}.bin")
      @entropy_path = File.join(@tmp_dir, "imm-horizon-entropy-#{stamp}.bin")
      @max_target_path = File.join(@tmp_dir, "imm-horizon-max-target-#{stamp}.bin")
      @max_mass_path = File.join(@tmp_dir, "imm-horizon-max-mass-#{stamp}.bin")
      @target_count_path = File.join(@tmp_dir, "imm-horizon-target-count-#{stamp}.bin")
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

        node_set = load_node_set(db)
        initialize_array_files(node_set.ids.size)

        @directions.each do |direction|
          source = direction == REVERSE ? reversed(corpus) : corpus
          build_direction(source, direction, max_depth, node_set)
        end

        write_table(db, node_set)
        write_meta(db, corpus.size, max_depth, node_set.ids.size)
      ensure
        cleanup_array_files
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
      raise BuildError.new("horizon arrays require a DB built with --schema id") unless schema == "id"
    end

    private def load_node_set(db : DB::Database) : NodeSet
      direction_sql = @directions.join(",")
      ids = [] of Int64
      row_by_id = Hash(Int64, Int32).new
      id_by_key = Hash(String, Int64).new

      STDERR.puts "loading node ids"
      db.query "SELECT id, direction, hex(key) FROM nodes WHERE direction IN (#{direction_sql}) ORDER BY id" do |rs|
        rs.each do
          id = rs.read(Int64)
          direction = rs.read(Int64).to_i
          key_hex = rs.read(String)
          row_by_id[id] = ids.size
          ids << id
          id_by_key["#{direction}:#{key_hex}"] = id
        end
      end

      NodeSet.new(ids, row_by_id, id_by_key)
    end

    private def initialize_array_files(node_count : Int32) : Nil
      FileUtils.mkdir_p(@tmp_dir)
      cell_count = node_count.to_i64 * @horizon
      truncate(@total_path, cell_count * 4)
      truncate(@entropy_path, cell_count * 4)
      truncate(@max_target_path, cell_count)
      truncate(@max_mass_path, cell_count * 4)
      truncate(@target_count_path, cell_count * 2)
    end

    private def truncate(path : String, size : Int64) : Nil
      File.open(path, "w") { |file| file.truncate(size) }
    end

    private def build_direction(corpus : Bytes, direction : Int32, max_depth : Int32, node_set : NodeSet) : Nil
      label = direction == FORWARD ? "forward" : "reverse"
      start_limit = corpus.size - max_depth - @horizon + 1
      hex = hex_table

      h = 1
      while h <= @horizon
        STDERR.puts "counting #{label} horizon #{h}/#{@horizon}"
        counts = Hash(UInt64, Int32).new(0)
        observations = 0_i64

        i = 0
        while i < start_limit
          key_hex = String.new
          depth = 1
          while depth <= max_depth
            token = corpus[i + depth - 1]
            key_hex += hex[token]
            node_id = node_set.id_by_key["#{direction}:#{key_hex}"]?
            raise BuildError.new("missing node id for #{label} key #{key_hex}") unless node_id

            target = corpus[i + depth - 1 + h]
            counts[pack_count_key(node_id, target)] += 1
            observations += 1
            depth += 1
          end
          i += 1
        end

        write_horizon_summary(h, counts, node_set)
        STDERR.puts "summarized #{label} horizon #{h}: #{observations} observations, #{counts.size} cells"
        h += 1
      end
    end

    private def write_horizon_summary(horizon : Int32, counts : Hash(UInt64, Int32), node_set : NodeSet) : Nil
      totals = Hash(Int64, Int32).new(0)
      target_counts = Hash(Int64, Int32).new(0)
      max_targets = Hash(Int64, UInt8).new(0_u8)
      max_masses = Hash(Int64, Int32).new(0)

      counts.each do |packed, mass|
        node_id, target = unpack_count_key(packed)
        totals[node_id] += mass
        target_counts[node_id] += 1
        if mass > max_masses[node_id]
          max_masses[node_id] = mass
          max_targets[node_id] = target
        end
      end

      entropy_acc = Hash(Int64, Float64).new(0.0)
      counts.each do |packed, mass|
        node_id, _target = unpack_count_key(packed)
        total = totals[node_id]
        p = mass.to_f64 / total
        entropy_acc[node_id] -= p * Math.log(p)
      end

      File.open(@total_path, "r+") do |total_file|
        File.open(@entropy_path, "r+") do |entropy_file|
          File.open(@max_target_path, "r+") do |max_target_file|
            File.open(@max_mass_path, "r+") do |max_mass_file|
              File.open(@target_count_path, "r+") do |target_count_file|
                totals.each do |node_id, total|
                  row = node_set.row_by_id[node_id]
                  cell = row.to_i64 * @horizon + (horizon - 1)
                  write_u32(total_file, cell * 4, total)
                  write_f32(entropy_file, cell * 4, entropy_acc[node_id].to_f32)
                  write_u8(max_target_file, cell, max_targets[node_id])
                  write_u32(max_mass_file, cell * 4, max_masses[node_id])
                  write_u16(target_count_file, cell * 2, target_counts[node_id])
                end
              end
            end
          end
        end
      end
    end

    private def write_table(db : DB::Database, node_set : NodeSet) : Nil
      existing = db.query_one?("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'horizon_arrays'", as: Int64) || 0_i64
      if existing > 0
        rows = db.query_one("SELECT COUNT(*) FROM horizon_arrays", as: Int64)
        if rows > 0 && !@replace
          raise BuildError.new("horizon_arrays already has #{rows} rows (use --replace)")
        end
        db.exec "DROP TABLE horizon_arrays"
      end

      db.exec <<-SQL
        CREATE TABLE horizon_arrays (
          node_id INTEGER NOT NULL PRIMARY KEY,
          max_horizon INTEGER NOT NULL,
          total_mass BLOB NOT NULL,
          entropy BLOB NOT NULL,
          max_target BLOB NOT NULL,
          max_mass BLOB NOT NULL,
          target_count BLOB NOT NULL
        ) WITHOUT ROWID
      SQL

      total_file = File.open(@total_path, "r")
      entropy_file = File.open(@entropy_path, "r")
      max_target_file = File.open(@max_target_path, "r")
      max_mass_file = File.open(@max_mass_path, "r")
      target_count_file = File.open(@target_count_path, "r")

      begin
        db.exec "BEGIN"
        node_set.ids.each_with_index do |node_id, row|
          total = read_blob(total_file, row, 4)
          entropy = read_blob(entropy_file, row, 4)
          max_target = read_blob(max_target_file, row, 1)
          max_mass = read_blob(max_mass_file, row, 4)
          target_count = read_blob(target_count_file, row, 2)
          db.exec <<-SQL, node_id, @horizon, total, entropy, max_target, max_mass, target_count
            INSERT INTO horizon_arrays(node_id, max_horizon, total_mass, entropy, max_target, max_mass, target_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
          SQL
        end
        db.exec "COMMIT"
      rescue ex
        db.exec "ROLLBACK" rescue nil
        raise ex
      ensure
        total_file.close
        entropy_file.close
        max_target_file.close
        max_mass_file.close
        target_count_file.close
      end
    end

    private def read_blob(file : File, row : Int32, width : Int32) : Bytes
      size = @horizon * width
      bytes = Bytes.new(size)
      file.seek(row.to_i64 * size)
      file.read_fully(bytes)
      bytes
    end

    private def write_meta(db : DB::Database, corpus_bytes : Int32, max_depth : Int32, node_count : Int32) : Nil
      horizon_starts = corpus_bytes - max_depth - @horizon + 1
      entries = {
        "horizon_array_max" => @horizon.to_s,
        "horizon_array_recorded_starts" => horizon_starts.to_s,
        "horizon_array_directions" => @directions.sort.map { |d| d == FORWARD ? "forward" : "reverse" }.join(","),
        "horizon_array_nodes" => node_count.to_s,
      }

      entries.each do |key, value|
        db.exec <<-SQL, key, value
          INSERT INTO meta(key, value) VALUES (?, ?)
          ON CONFLICT(key) DO UPDATE SET value = excluded.value
        SQL
      end
    end

    private def write_u8(file : File, offset : Int64, value : UInt8) : Nil
      bytes = Bytes[value]
      file.seek(offset)
      file.write(bytes)
    end

    private def write_u16(file : File, offset : Int64, value : Int32) : Nil
      v = value.to_u16
      bytes = Bytes[(v & 0xff).to_u8, ((v >> 8) & 0xff).to_u8]
      file.seek(offset)
      file.write(bytes)
    end

    private def write_u32(file : File, offset : Int64, value : Int32) : Nil
      v = value.to_u32
      bytes = Bytes[
        (v & 0xff).to_u8,
        ((v >> 8) & 0xff).to_u8,
        ((v >> 16) & 0xff).to_u8,
        ((v >> 24) & 0xff).to_u8,
      ]
      file.seek(offset)
      file.write(bytes)
    end

    private def write_f32(file : File, offset : Int64, value : Float32) : Nil
      bits = value.unsafe_as(UInt32)
      bytes = Bytes[
        (bits & 0xff).to_u8,
        ((bits >> 8) & 0xff).to_u8,
        ((bits >> 16) & 0xff).to_u8,
        ((bits >> 24) & 0xff).to_u8,
      ]
      file.seek(offset)
      file.write(bytes)
    end

    private def meta_string(db : DB::Database, key : String) : String
      value = db.query_one?("SELECT value FROM meta WHERE key = ?", key, as: String)
      raise BuildError.new("missing meta key: #{key}") unless value
      value
    end

    private def meta_int(db : DB::Database, key : String) : Int32
      meta_string(db, key).to_i
    end

    private def pack_count_key(node_id : Int64, target : UInt8) : UInt64
      (node_id.to_u64 << 8) | target.to_u64
    end

    private def unpack_count_key(value : UInt64) : Tuple(Int64, UInt8)
      {(value >> 8).to_i64, (value & 0xff_u64).to_u8}
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

    private def cleanup_array_files : Nil
      {@total_path, @entropy_path, @max_target_path, @max_mass_path, @target_count_path}.each do |path|
        File.delete(path) if File.exists?(path)
      end
    end
  end
end

db_path = nil
input_path = nil
horizon = 16
replace = false
direction_arg = "forward"
tmp_dir = "/tmp"

parser = OptionParser.new do |p|
  p.banner = "Usage: imm_horizon_array_build --db PATH --input PATH --horizon N [--replace]"

  p.on("--db=PATH", "ID-schema SQLite tree database to augment") { |path| db_path = path }
  p.on("-i PATH", "--input=PATH", "Corpus text file used to build the tree") { |path| input_path = path }
  p.on("--horizon=N", "Maximum future offset to summarize") { |n| horizon = n.to_i }
  p.on("--direction=NAME", "forward, reverse, or both (default: forward)") { |name| direction_arg = name }
  p.on("--tmp-dir=PATH", "Temporary directory for array files") { |path| tmp_dir = path }
  p.on("--replace", "Replace existing horizon_arrays table") { replace = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

begin
  parser.parse
  raise IMMHorizonArray::BuildError.new("--db is required") unless db_path
  raise IMMHorizonArray::BuildError.new("--input is required") unless input_path

  directions = case direction_arg
               when "forward"
                 [IMMHorizonArray::FORWARD]
               when "reverse"
                 [IMMHorizonArray::REVERSE]
               when "both"
                 [IMMHorizonArray::FORWARD, IMMHorizonArray::REVERSE]
               else
                 raise IMMHorizonArray::BuildError.new("--direction must be forward, reverse, or both")
               end

  IMMHorizonArray::Builder.new(
    db_path.not_nil!,
    input_path.not_nil!,
    horizon,
    replace,
    directions,
    tmp_dir
  ).run
rescue ex : IMMHorizonArray::BuildError
  STDERR.puts "error: #{ex.message}"
  STDERR.puts parser
  exit 2
end
