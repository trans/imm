.headers on
.mode column

-- Materialize per-context entropy stats for each direction.
CREATE TEMP TABLE context_stats AS
WITH dist AS (
  SELECT
    direction,
    parent_key AS key,
    length(parent_key) AS depth,
    token,
    mass,
    SUM(mass) OVER (PARTITION BY direction, parent_key) AS total_mass
  FROM nodes
)
SELECT
  direction,
  key,
  depth,
  total_mass AS mass,
  COUNT(*) AS branching_factor,
  -SUM((CAST(mass AS REAL) / total_mass) * ln(CAST(mass AS REAL) / total_mass)) AS entropy
FROM dist
GROUP BY direction, key;

CREATE INDEX context_stats_dir_key_idx ON context_stats(direction, key);

-- SQLite has no built-in reverse(blob), so this first query compares summary
-- distributions by depth and mass. For specific keys use pin_pout_example.sql.
SELECT
  CASE direction WHEN 0 THEN 'forward P_out' ELSE 'reverse P_in' END AS side,
  depth,
  COUNT(*) AS contexts,
  SUM(CASE WHEN branching_factor = 1 THEN 1 ELSE 0 END) AS zero_entropy_contexts,
  ROUND(AVG(entropy), 6) AS avg_entropy,
  ROUND(AVG(mass), 3) AS avg_mass,
  MAX(branching_factor) AS max_branching_factor
FROM context_stats
GROUP BY direction, depth
ORDER BY direction, depth;

SELECT
  CASE direction WHEN 0 THEN 'forward P_out' ELSE 'reverse P_in' END AS side,
  depth,
  quote(key) AS key,
  mass,
  branching_factor,
  ROUND(entropy, 6) AS entropy
FROM context_stats
WHERE depth > 0
ORDER BY entropy ASC, mass DESC
LIMIT 100;
