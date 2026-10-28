# v11b shards (= v11; reddit-a 9000 s / $34 so v11b + the 1,786 s and $0.00036 used by v11 attempts 392750-57 fit the approved 104,400 s / $280)
# v11 shards: NAME SITE SECONDS API_USD WORKERS TASKS; each task runs no_search then piotr_selectionarm (interleaved).
# 128 tasks = Shopping 45 + CMS 35 + Reddit 18 + GitLab 30; 256 cells.
V11B_SHARDS=(
"shop-a shopping 14400 35 2 4,10,22,28,29,31,36,37,41,42,46,54,55,56,58"
"shop-b shopping 14400 35 2 65,66,67,68,72,74,76,77,78,79,82,84,86,89,93"
"shop-c shopping 14400 35 2 94,95,103,104,109,117,118,119,120,121,123,129,142,148,162"
"cms-a shopping_admin 10800 35 2 0,2,8,13,17,21,26,27,30,32,33,40,47,48,50,51,59,70"
"cms-b shopping_admin 10800 35 2 80,88,101,106,107,108,110,113,114,115,116,125,126,145,149,150,160"
"reddit-a reddit 9000 34 2 14,97,98,131,132,133,134,135,136,137,138,139,140,141,151,152,153,154"
"gitlab-a gitlab 14400 35 2 9,25,34,39,43,44,45,49,62,63,71,73,75,81,83"
"gitlab-b gitlab 14400 35 2 96,99,100,105,111,112,122,124,128,130,143,156,157,163,164"
)
