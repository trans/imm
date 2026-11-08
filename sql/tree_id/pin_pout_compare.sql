.headers on
.mode column

-- ID-schema version of P_out(. | K) versus P_in(. | K).
-- Child distributions are grouped by parent_id. Keys are only used to match
-- forward context K against reverse context reverse(K).

CREATE TEMP TABLE context_stats AS
WITH dist AS (
  SELECT
    child.direction,
    child.parent_id AS context_id,
    child.token,
    child.mass,
    SUM(child.mass) OVER (PARTITION BY child.direction, child.parent_id) AS total_mass
  FROM nodes AS child
),
stats AS (
  SELECT
    dist.direction,
    dist.context_id,
    CASE
      WHEN dist.context_id = 0 THEN ''
      ELSE hex(parent.key)
    END AS key_hex,
    CASE
      WHEN dist.context_id = 0 THEN 0
      ELSE parent.depth
    END AS depth,
    dist.total_mass AS mass,
    COUNT(*) AS branching_factor,
    -SUM((CAST(dist.mass AS REAL) / dist.total_mass) * ln(CAST(dist.mass AS REAL) / dist.total_mass)) AS entropy
  FROM dist
  LEFT JOIN nodes AS parent
    ON parent.id = dist.context_id
  GROUP BY dist.direction, dist.context_id
)
SELECT
  direction,
  context_id,
  key_hex,
  CASE length(key_hex)
    WHEN 0 THEN ''
    WHEN 2 THEN key_hex
    WHEN 4 THEN substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 6 THEN substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 8 THEN substr(key_hex, 7, 2) || substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 10 THEN substr(key_hex, 9, 2) || substr(key_hex, 7, 2) || substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 12 THEN substr(key_hex, 11, 2) || substr(key_hex, 9, 2) || substr(key_hex, 7, 2) || substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 14 THEN substr(key_hex, 13, 2) || substr(key_hex, 11, 2) || substr(key_hex, 9, 2) || substr(key_hex, 7, 2) || substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
    WHEN 16 THEN substr(key_hex, 15, 2) || substr(key_hex, 13, 2) || substr(key_hex, 11, 2) || substr(key_hex, 9, 2) || substr(key_hex, 7, 2) || substr(key_hex, 5, 2) || substr(key_hex, 3, 2) || substr(key_hex, 1, 2)
  END AS reverse_key_hex,
  depth,
  mass,
  branching_factor,
  entropy
FROM stats;

CREATE INDEX context_stats_dir_key_idx ON context_stats(direction, key_hex);

CREATE TEMP TABLE pin_pout AS
SELECT
  f.key_hex,
  f.depth,
  f.mass AS out_mass,
  r.mass AS in_mass,
  f.branching_factor AS out_branching,
  r.branching_factor AS in_branching,
  f.entropy AS out_entropy,
  r.entropy AS in_entropy,
  r.entropy - f.entropy AS in_minus_out_entropy,
  MIN(f.mass, r.mass) * ABS(r.entropy - f.entropy) AS weighted_entropy_gap
FROM context_stats f
JOIN context_stats r
  ON r.direction = 1
 AND r.key_hex = f.reverse_key_hex
WHERE f.direction = 0;

SELECT
  depth,
  COUNT(*) AS contexts,
  ROUND(AVG(out_entropy), 6) AS avg_out_entropy,
  ROUND(AVG(in_entropy), 6) AS avg_in_entropy,
  ROUND(AVG(in_minus_out_entropy), 6) AS avg_in_minus_out,
  ROUND(AVG(weighted_entropy_gap), 3) AS avg_weighted_gap
FROM pin_pout
GROUP BY depth
ORDER BY depth;

SELECT
  'incoming << outgoing' AS regulator,
  depth,
  CAST(unhex(key_hex) AS TEXT) AS key_text,
  key_hex,
  out_mass,
  in_mass,
  out_branching,
  in_branching,
  ROUND(out_entropy, 6) AS out_entropy,
  ROUND(in_entropy, 6) AS in_entropy,
  ROUND(weighted_entropy_gap, 3) AS weighted_gap
FROM pin_pout
WHERE out_mass >= 10
  AND in_mass >= 10
  AND in_minus_out_entropy < 0
ORDER BY weighted_entropy_gap DESC, in_minus_out_entropy ASC
LIMIT 50;

SELECT
  'incoming >> outgoing' AS regulator,
  depth,
  CAST(unhex(key_hex) AS TEXT) AS key_text,
  key_hex,
  out_mass,
  in_mass,
  out_branching,
  in_branching,
  ROUND(out_entropy, 6) AS out_entropy,
  ROUND(in_entropy, 6) AS in_entropy,
  ROUND(weighted_entropy_gap, 3) AS weighted_gap
FROM pin_pout
WHERE out_mass >= 10
  AND in_mass >= 10
  AND in_minus_out_entropy > 0
ORDER BY weighted_entropy_gap DESC, in_minus_out_entropy DESC
LIMIT 50;
