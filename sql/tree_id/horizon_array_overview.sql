.headers on
.mode column

SELECT key, value
FROM meta
WHERE key LIKE 'horizon_array_%'
ORDER BY key;

SELECT
  COUNT(*) AS nodes,
  MIN(max_horizon) AS min_horizon,
  MAX(max_horizon) AS max_horizon,
  SUM(length(total_mass) + length(entropy) + length(max_target) + length(max_mass) + length(target_count)) AS payload_bytes,
  ROUND(AVG(length(total_mass) + length(entropy) + length(max_target) + length(max_mass) + length(target_count)), 2) AS avg_payload_bytes
FROM horizon_arrays;
