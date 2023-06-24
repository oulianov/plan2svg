# Plan 2 SVG

Experiments on how to convert a building floor plan into a vectorized representation of its walls and entries. 

Input is a bitmap picture of a floor plan. It may be annotated, dirty, noisy, etc. 

Output is a set of walls and entries. 
- Walls are segments, eventually curved. 
- Entries are 'spaces' in between two walls. 