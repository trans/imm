.headers on
.mode column

-- Rank node/horizon pairs with sharp future-token distributions.
CREATE TEMP TABLE horizon_stats AS
WITH dist AS (
  SELECT
    node_id,
    horizon,
    target,
    mass,
    SUM(mass) OVER (PARTITION BY node_id, horizon) AS total_mass
  FROM horizon_counts
)
SELECT
  node_id,
  horizon,
  total_mass AS mass,
  COUNT(*) AS target_count,
  MAX(CAST(mass AS REAL) / total_mass) AS max_p,
  -SUM((CAST(mass AS REAL) / total_mass) * ln(CAST(mass AS REAL) / total_mass)) AS entropy
FROM dist
GROUP BY node_id, horizon;

SELECT
  hs.horizon,
  n.depth,
  CAST(n.key AS TEXT) AS key_text,
  hex(n.key) AS key_hex,
  hs.mass,
  hs.target_count,
  ROUND(hs.max_p, 6) AS max_p,
  ROUND(hs.entropy, 6) AS entropy,
  ROUND(hs.mass * (1.0 - hs.entropy / ln(256.0)), 3) AS weighted_sharpness
FROM horizon_stats hs
JOIN nodes n ON n.id = hs.node_id
WHERE hs.mass >= 10
ORDER BY weighted_sharpness DESC, hs.mass DESC
LIMIT 100;
