GRID* 1 0 0
* 0
GRID* 2 0 0
* 1
GRID* 3 0 0
* 2
GRID* 4 0 1
* 0
GRID* 5 0 1
* 1
GRID* 6 0 1
* 2
GRID* 7 0 2
* 0
GRID* 8 0 2
* 1
GRID* 9 0 2
* 2
GRID* 10 1 0
* 0
GRID* 11 1 0
* 1
GRID* 12 1 0
* 2
GRID* 13 1 1
* 0
GRID* 14 1 1
* 1
GRID* 15 1 1
* 2
GRID* 16 1 2
* 0
GRID* 17 1 2
* 1
GRID* 18 1 2
* 2
GRID* 19 2 0
* 0
GRID* 20 2 0
* 1
GRID* 21 2 0
* 2
GRID* 22 2 1
* 0
GRID* 23 2 1
* 1
GRID* 24 2 1
* 2
GRID* 25 2 2
* 0
GRID* 26 2 2
* 1
GRID* 27 2 2
* 2
CHEXA 1 1 1 2 5 4 10 11+ 1
+ 1 14 13
CHEXA 2 1 2 3 6 5 11 12+ 2
+ 2 15 14
CHEXA 3 1 4 5 8 7 13 14+ 3
+ 3 17 16
CHEXA 4 1 5 6 9 8 14 15+ 4
+ 4 18 17
CHEXA 5 1 10 11 14 13 19 20+ 5
+ 5 23 22
CHEXA 6 1 11 12 15 14 20 21+ 6
+ 6 24 23
CHEXA 7 1 13 14 17 16 22 23+ 7
+ 7 26 25
CHEXA 8 1 14 15 18 17 23 24+ 8
+ 8 27 26
$HMNAME COMP                1000"PSOLID_1000"
SPC         5302       1     123     0.0
SPC         5302       2     123     0.0
SPC         5302       3     123     0.0
SPC         5302       4     123     0.0
SPC         5302       5     123     0.0
SPC         5302       6     123     0.0
SPC         5302       7     123     0.0
SPC         5302       8     123     0.0
SPC         5302       9     123     0.0
FORCE*                 1               1               0         1.00000
*                1.00000         0.00000         0.00000
FORCE*                 2               1               0         1.00000
*                0.00000         1.00000         0.00000
FORCE*                 3               1               0         1.00000
*                0.00000         0.00000         1.00000
ENDDATA