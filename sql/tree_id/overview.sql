.headers on
.mode column

SELECT key, value
FROM meta
ORDER BY key;

SELECT
  CASE direction WHEN 0 THEN 'forward' ELSE 'reverse' END AS direction,
  depth,
  COUNT(*) AS nodes,
  SUM(mass) AS total_mass,
  SUM(CASE WHEN mass = 1 THEN 1 ELSE 0 END) AS singletons,
  ROUND(100.0 * SUM(CASE WHEN mass = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS singleton_pct,
  MIN(mass) AS min_mass,
  MAX(mass) AS max_mass,
  ROUND(AVG(mass), 3) AS avg_mass
FROM nodes
GROUP BY direction, depth
ORDER BY direction, depth;

SELECT
  token,
  printf('%02X', token) AS hex,
  CASE
    WHEN token = 10 THEN '\\n'
    WHEN token = 13 THEN '\\r'
    WHEN token = 9 THEN '\\t'
    WHEN token BETWEEN 32 AND 126 THEN char(token)
    ELSE ''
  END AS text,
  mass
FROM nodes
WHERE direction = 0
  AND parent_id = 0
ORDER BY mass DESC, token
LIMIT 40;
