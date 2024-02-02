"""
For each block, store the type of block (first, second):
- first=0: FEM tet
- first=1: ABD affine body
================= Contact blocks =================
- first=2: PG (point-ground) contact. Actually a point-plane contact, 
    but the letter P is already used for point-point contact.
- first=3: PT (point-triangle) contact, x0-(x1, x2, x3)
    - second=0: pt(x0, x1, x2, x3)
    - second=1: pp(x0, x1)
    - second=2: pp(x0, x2)
    - second=3: pe(x0, x1, x2)
    - second=4: pp(x0, x3)
    - second=5: pe(x0, x3, x1)
    - second=6: pe(x0, x2, x3)
- first=4: EE (edge-edge) contact, (x0, x1)-(x2, x3)
    - second=0: pp (x0, x2)
    - second=1: pp (x0, x3)
    - second=2: pp (x1, x2)
    - second=3: pp (x1, x3)
    - second=4: pe (x0, x2, x3)
    - second=5: pe (x1, x2, x3)
    - second=6: pe (x2, x0, x1)
    - second=7: pe (x3, x0, x1)
    - second=8: ee (x0, x1, x2, x3)
- first=5 (PP), first=6 (PE): not used
- first=7: constraint block
"""

AFFINE_NONE = -1
AFFINE_PROXY = -2

FEM_BLOCK = 0
ABD_AFFINE_BLOCK = 1
PG_CONTACT_BLOCK = 2
PT_CONTACT_BLOCK = 3
EE_CONTACT_BLOCK = 4
PP_CONTACT_BLOCK = 5
PE_CONTACT_BLOCK = 6
CONSTRAINT_BLOCK = 7

FEM_TET_BLOCK = 0
FEM_TRI_BLOCK = 1
FEM_EDGE_BLOCK = 2  # not used
FEM_HINGE_BLOCK = 3

PT_CONTACT_PT = 0
PT_CONTACT_PP01 = 1
PT_CONTACT_PP02 = 2
PT_CONTACT_PE012 = 3
PT_CONTACT_PP03 = 4
PT_CONTACT_PE031 = 5
PT_CONTACT_PE023 = 6

EE_CONTACT_PP02 = 0
EE_CONTACT_PP03 = 1
EE_CONTACT_PP12 = 2
EE_CONTACT_PP13 = 3
EE_CONTACT_PE023 = 4
EE_CONTACT_PE123 = 5
EE_CONTACT_PE201 = 6
EE_CONTACT_PE301 = 7
EE_CONTACT_EE = 8

CONSTRAINT_EQ1 = 0  # ||x0 - target0|| = 0
CONSTRAINT_EQ2 = 1  # ||x0 - x1|| = 0
CONSTRAINT_ANGULAR_PID_CONTROL = 2  # Angular PID 
CONSTRAINT_AREA = 3  # Area constraint
CONSTRAINT_DIST_SYM = 4  # Symmetric distance constraint for prismatic gripper, ||x1-x0+x3-x2|| = 0
CONSTRAINT_DIST_PID_CONTROL = 5  # Distance PID control on ||x0 - x1||
