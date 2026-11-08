.headers on
.mode column

SELECT
  CASE direction WHEN 0 THEN 'forward' ELSE 'reverse' END AS direction,
  length(parent_key) + 1 AS depth,
  quote(parent_key) AS parent_key,
  token,
  printf('%02X', token) AS token_hex,
  CASE
    WHEN token = 10 THEN '\\n'
    WHEN token = 13 THEN '\\r'
    WHEN token = 9 THEN '\\t'
    WHEN token BETWEEN 32 AND 126 THEN char(token)
    ELSE ''
  END AS token_text,
  mass
FROM nodes
ORDER BY mass DESC, depth, parent_key, token
LIMIT 100;
