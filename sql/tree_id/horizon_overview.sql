.headers on
.mode column

SELECT key, value
FROM meta
WHERE key LIKE 'horizon_%'
ORDER BY key;

SELECT
  horizon,
  COUNT(*) AS rows,
  SUM(mass) AS total_mass,
  COUNT(DISTINCT node_id) AS nodes,
  ROUND(AVG(mass), 3) AS avg_cell_mass,
  MAX(mass) AS max_cell_mass
FROM horizon_counts
GROUP BY horizon
ORDER BY horizon;

SELECT
  hc.horizon,
  n.depth,
  COUNT(*) AS rows,
  SUM(hc.mass) AS total_mass,
  COUNT(DISTINCT hc.node_id) AS nodes
FROM horizon_counts hc
JOIN nodes n ON n.id = hc.node_id
GROUP BY hc.horizon, n.depth
ORDER BY hc.horizon, n.depth;
