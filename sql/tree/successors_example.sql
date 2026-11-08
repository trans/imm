.headers on
.mode column

-- Edit this CTE to inspect a context. The builder currently stores raw bytes,
-- so ordinary ASCII text can be cast directly to BLOB.
WITH
  input(ctx) AS (VALUES(CAST('th' AS BLOB))),
  total(n) AS (
    SELECT SUM(mass)
    FROM nodes, input
    WHERE direction = 0
      AND parent_key = input.ctx
  )
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
  mass,
  ROUND(CAST(mass AS REAL) / total.n, 6) AS p
FROM nodes, input, total
WHERE direction = 0
  AND parent_key = input.ctx
ORDER BY mass DESC, token
LIMIT 80;
