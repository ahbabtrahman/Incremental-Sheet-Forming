
task -slotname gsictrlts -slotid 119 -pri 5 -vwopt 0x1c -stcks 8000 \
-entp gsictrlts_main -auto


task -slotname udpuc -slotid 138 -pri 77 -vwopt 0x1c -stcks 20000 \
-entp udpuc_main -auto -noreg
