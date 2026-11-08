#!/usr/bin/env crystal

require "db"
require "option_parser"
require "sqlite3"

module IMMHorizonProfile
  class ProfileError < Exception
  end

  struct Row
    getter key : Bytes
    getter depth : Int32
    getter node_mass : Int64
    getter best_horizon : Int32
    getter local_entropy : Float64
    getter best_entropy : Float64
    getter best_max_p : Float64
    getter best_target : UInt8
    getter best_total : UInt32
    getter best_target_count : UInt16
    getter entropy_drop : Float64
    getter peak_sharpness : Float64
    getter area_sharpness : Float64
    getter score : Float64

    def initialize(
      @key,
      @depth,
      @node_mass,
      @best_horizon,
      @local_entropy,
      @best_entropy,
      @best_max_p,
      @best_target,
      @best_total,
      @best_target_count,
      @entropy_drop,
      @peak_sharpness,
      @area_sharpness,
      @score
    )
    end
  end

  class Runner
    getter db_path : String
    getter limit : Int32
    getter min_mass : Int32
    getter min_depth : Int32
    getter max_depth : Int32?
    getter late_only : Bool
    getter min_best_horizon : Int32

    def initialize(@db_path, @limit, @min_mass, @min_depth, @max_depth, @late_only, @min_best_horizon)
      raise ProfileError.new("database does not exist: #{@db_path}") unless File.file?(@db_path)
      raise ProfileError.new("--limit must be positive") unless @limit > 0
      raise ProfileError.new("--min-mass must be non-negative") unless @min_mass >= 0
      raise ProfileError.new("--min-best-h must be positive") unless @min_best_horizon > 0
    end

    def run : Nil
      rows = [] of Row
      DB.open("sqlite3://#{File.expand_path(@db_path)}") do |db|
        root_entropy = root_entropy(db)
        max_horizon = horizon_max(db)
        query = String.build do |sql|
          sql << "SELECT n.key, n.depth, n.mass, ha.total_mass, ha.entropy, ha.max_target, ha.max_mass, ha.target_count "
          sql << "FROM horizon_arrays ha JOIN nodes n ON n.id = ha.node_id "
          sql << "WHERE n.mass >= ? AND n.depth >= ?"
          sql << " AND n.depth <= ?" if @max_depth
        end
        args = [] of DB::Any
        args << @min_mass
        args << @min_depth
        args << @max_depth.not_nil! if @max_depth

        db.query query, args: args do |rs|
          rs.each do
            row = profile_row(rs, root_entropy, max_horizon)
            next if @late_only && row.best_horizon <= 1
            next if row.best_horizon < @min_best_horizon
            rows << row
          end
        end
      end

      rows.sort_by! { |row| {-row.score, -row.node_mass, row.depth, printable(row.key)} }
      print_rows(rows.first(@limit))
    end

    private def profile_row(rs : DB::ResultSet, root_entropy : Float64, max_horizon : Int32) : Row
      key = rs.read(Bytes)
      depth = rs.read(Int64).to_i
      node_mass = rs.read(Int64)
      total_blob = rs.read(Bytes)
      entropy_blob = rs.read(Bytes)
      max_target_blob = rs.read(Bytes)
      max_mass_blob = rs.read(Bytes)
      target_count_blob = rs.read(Bytes)

      local_entropy = read_f32(entropy_blob, 0).to_f64
      best_h = 1
      best_entropy = local_entropy
      best_max_p = 0.0
      best_target = 0_u8
      best_total = 0_u32
      best_target_count = 0_u16
      peak_sharpness = 0.0
      area_sharpness = 0.0

      h = 1
      while h <= max_horizon
        idx = h - 1
        total = read_u32(total_blob, idx)
        entropy = read_f32(entropy_blob, idx).to_f64
        max_mass = read_u32(max_mass_blob, idx)
        max_p = total == 0 ? 0.0 : max_mass.to_f64 / total
        sharpness = total == 0 ? 0.0 : 1.0 - entropy / root_entropy
        area_sharpness += sharpness

        if sharpness > peak_sharpness || (sharpness == peak_sharpness && max_p > best_max_p)
          peak_sharpness = sharpness
          best_h = h
          best_entropy = entropy
          best_max_p = max_p
          best_target = max_target_blob[idx]
          best_total = total
          best_target_count = read_u16(target_count_blob, idx)
        end

        h += 1
      end

      entropy_drop = local_entropy - best_entropy
      late_bonus = best_h <= 1 ? 0.5 : 1.0
      score = Math.log1p(node_mass.to_f64) * [local_entropy, 0.05].max * [peak_sharpness, 0.0].max * late_bonus

      Row.new(
        key,
        depth,
        node_mass,
        best_h,
        local_entropy,
        best_entropy,
        best_max_p,
        best_target,
        best_total,
        best_target_count,
        entropy_drop,
        peak_sharpness,
        area_sharpness,
        score
      )
    end

    private def root_entropy(db : DB::Database) : Float64
      value = db.query_one? <<-SQL, as: Float64
        WITH dist AS (
          SELECT mass, SUM(mass) OVER () AS total_mass
          FROM nodes
          WHERE direction = 0 AND parent_id = 0
        )
        SELECT -SUM((CAST(mass AS REAL) / total_mass) * ln(CAST(mass AS REAL) / total_mass))
        FROM dist
      SQL
      value || Math.log(256.0)
    end

    private def horizon_max(db : DB::Database) : Int32
      value = db.query_one?("SELECT MAX(max_horizon) FROM horizon_arrays", as: Int64)
      raise ProfileError.new("horizon_arrays is empty") unless value
      value.to_i
    end

    private def print_rows(rows : Array(Row)) : Nil
      puts [
        "rank",
        "score",
        "key",
        "hex",
        "depth",
        "mass",
        "best_h",
        "local_H",
        "best_H",
        "H_drop",
        "max_p",
        "target",
        "target_count",
        "peak_sharp",
        "area_sharp",
      ].join('\t')

      rows.each_with_index(1) do |row, rank|
        puts [
          rank,
          fmt(row.score),
          printable(row.key),
          row.key.hexstring.upcase,
          row.depth,
          row.node_mass,
          row.best_horizon,
          fmt(row.local_entropy),
          fmt(row.best_entropy),
          fmt(row.entropy_drop),
          fmt(row.best_max_p),
          token_text(row.best_target),
          row.best_target_count,
          fmt(row.peak_sharpness),
          fmt(row.area_sharpness),
        ].join('\t')
      end
    end

    private def read_u16(bytes : Bytes, index : Int32) : UInt16
      offset = index * 2
      (bytes[offset].to_u16 |
        (bytes[offset + 1].to_u16 << 8))
    end

    private def read_u32(bytes : Bytes, index : Int32) : UInt32
      offset = index * 4
      (bytes[offset].to_u32 |
        (bytes[offset + 1].to_u32 << 8) |
        (bytes[offset + 2].to_u32 << 16) |
        (bytes[offset + 3].to_u32 << 24))
    end

    private def read_f32(bytes : Bytes, index : Int32) : Float32
      read_u32(bytes, index).unsafe_as(Float32)
    end

    private def printable(bytes : Bytes) : String
      String.build do |io|
        bytes.each do |byte|
          case byte
          when 9
            io << "\\t"
          when 10
            io << "\\n"
          when 13
            io << "\\r"
          when 32..126
            io << byte.chr
          else
            io << "\\x" << byte.to_s(16).rjust(2, '0').upcase
          end
        end
      end
    end

    private def token_text(byte : UInt8) : String
      case byte
      when 9
        "\\t"
      when 10
        "\\n"
      when 13
        "\\r"
      when 32..126
        byte.chr.to_s
      else
        "\\x#{byte.to_s(16).rjust(2, '0').upcase}"
      end
    end

    private def fmt(value : Float64) : String
      "%.6f" % value
    end
  end
end

db_path = nil
limit = 50
min_mass = 20
min_depth = 1
max_depth = nil
late_only = false
min_best_horizon = 1

parser = OptionParser.new do |p|
  p.banner = "Usage: imm_horizon_profile --db PATH [--limit N]"

  p.on("--db=PATH", "ID-schema SQLite DB with horizon_arrays") { |path| db_path = path }
  p.on("--limit=N", "Rows to print") { |n| limit = n.to_i }
  p.on("--min-mass=N", "Minimum node mass") { |n| min_mass = n.to_i }
  p.on("--min-depth=N", "Minimum node depth") { |n| min_depth = n.to_i }
  p.on("--max-depth=N", "Maximum node depth") { |n| max_depth = n.to_i }
  p.on("--late-only", "Only rows whose strongest horizon is h > 1") { late_only = true }
  p.on("--min-best-h=N", "Only rows whose strongest horizon is at least N") { |n| min_best_horizon = n.to_i }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

begin
  parser.parse
  raise IMMHorizonProfile::ProfileError.new("--db is required") unless db_path

  IMMHorizonProfile::Runner.new(
    db_path.not_nil!,
    limit,
    min_mass,
    min_depth,
    max_depth,
    late_only,
    min_best_horizon
  ).run
rescue ex : IMMHorizonProfile::ProfileError
  STDERR.puts "error: #{ex.message}"
  STDERR.puts parser
  exit 2
end
