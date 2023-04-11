\* Bin_Packing_Problem *\
Minimize
Sum_of_Containers_Used: BinUsed_0 + BinUsed_1 + BinUsed_2 + BinUsed_3
 + BinUsed_4 + BinUsed_5 + BinUsed_6 + BinUsed_7 + BinUsed_8 + BinUsed_9
Subject To
An_item_can_be_in_only_1_bin:_object_A: itemInBin_('A',_0)
 + itemInBin_('A',_1) + itemInBin_('A',_2) + itemInBin_('A',_3)
 + itemInBin_('A',_4) + itemInBin_('A',_5) + itemInBin_('A',_6)
 + itemInBin_('A',_7) + itemInBin_('A',_8) + itemInBin_('A',_9) = 1
An_item_can_be_in_only_1_bin:_object_B: itemInBin_('B',_0)
 + itemInBin_('B',_1) + itemInBin_('B',_2) + itemInBin_('B',_3)
 + itemInBin_('B',_4) + itemInBin_('B',_5) + itemInBin_('B',_6)
 + itemInBin_('B',_7) + itemInBin_('B',_8) + itemInBin_('B',_9) = 1
An_item_can_be_in_only_1_bin:_object_C: itemInBin_('C',_0)
 + itemInBin_('C',_1) + itemInBin_('C',_2) + itemInBin_('C',_3)
 + itemInBin_('C',_4) + itemInBin_('C',_5) + itemInBin_('C',_6)
 + itemInBin_('C',_7) + itemInBin_('C',_8) + itemInBin_('C',_9) = 1
An_item_can_be_in_only_1_bin:_object_D: itemInBin_('D',_0)
 + itemInBin_('D',_1) + itemInBin_('D',_2) + itemInBin_('D',_3)
 + itemInBin_('D',_4) + itemInBin_('D',_5) + itemInBin_('D',_6)
 + itemInBin_('D',_7) + itemInBin_('D',_8) + itemInBin_('D',_9) = 1
An_item_can_be_in_only_1_bin:_object_E: itemInBin_('E',_0)
 + itemInBin_('E',_1) + itemInBin_('E',_2) + itemInBin_('E',_3)
 + itemInBin_('E',_4) + itemInBin_('E',_5) + itemInBin_('E',_6)
 + itemInBin_('E',_7) + itemInBin_('E',_8) + itemInBin_('E',_9) = 1
An_item_can_be_in_only_1_bin:_object_F: itemInBin_('F',_0)
 + itemInBin_('F',_1) + itemInBin_('F',_2) + itemInBin_('F',_3)
 + itemInBin_('F',_4) + itemInBin_('F',_5) + itemInBin_('F',_6)
 + itemInBin_('F',_7) + itemInBin_('F',_8) + itemInBin_('F',_9) = 1
An_item_can_be_in_only_1_bin:_object_G: itemInBin_('G',_0)
 + itemInBin_('G',_1) + itemInBin_('G',_2) + itemInBin_('G',_3)
 + itemInBin_('G',_4) + itemInBin_('G',_5) + itemInBin_('G',_6)
 + itemInBin_('G',_7) + itemInBin_('G',_8) + itemInBin_('G',_9) = 1
An_item_can_be_in_only_1_bin:_object_H: itemInBin_('H',_0)
 + itemInBin_('H',_1) + itemInBin_('H',_2) + itemInBin_('H',_3)
 + itemInBin_('H',_4) + itemInBin_('H',_5) + itemInBin_('H',_6)
 + itemInBin_('H',_7) + itemInBin_('H',_8) + itemInBin_('H',_9) = 1
An_item_can_be_in_only_1_bin:_object_I: itemInBin_('I',_0)
 + itemInBin_('I',_1) + itemInBin_('I',_2) + itemInBin_('I',_3)
 + itemInBin_('I',_4) + itemInBin_('I',_5) + itemInBin_('I',_6)
 + itemInBin_('I',_7) + itemInBin_('I',_8) + itemInBin_('I',_9) = 1
An_item_can_be_in_only_1_bin:_object_J: itemInBin_('J',_0)
 + itemInBin_('J',_1) + itemInBin_('J',_2) + itemInBin_('J',_3)
 + itemInBin_('J',_4) + itemInBin_('J',_5) + itemInBin_('J',_6)
 + itemInBin_('J',_7) + itemInBin_('J',_8) + itemInBin_('J',_9) = 1
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_0:
 - 13 BinUsed_0 + 2 itemInBin_('A',_0) + 5 itemInBin_('B',_0)
 + 7 itemInBin_('C',_0) + 13 itemInBin_('D',_0) + itemInBin_('E',_0)
 + 4 itemInBin_('F',_0) + 9 itemInBin_('G',_0) + 2 itemInBin_('H',_0)
 + 7 itemInBin_('I',_0) + 6 itemInBin_('J',_0) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_1:
 - 13 BinUsed_1 + 2 itemInBin_('A',_1) + 5 itemInBin_('B',_1)
 + 7 itemInBin_('C',_1) + 13 itemInBin_('D',_1) + itemInBin_('E',_1)
 + 4 itemInBin_('F',_1) + 9 itemInBin_('G',_1) + 2 itemInBin_('H',_1)
 + 7 itemInBin_('I',_1) + 6 itemInBin_('J',_1) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_2:
 - 13 BinUsed_2 + 2 itemInBin_('A',_2) + 5 itemInBin_('B',_2)
 + 7 itemInBin_('C',_2) + 13 itemInBin_('D',_2) + itemInBin_('E',_2)
 + 4 itemInBin_('F',_2) + 9 itemInBin_('G',_2) + 2 itemInBin_('H',_2)
 + 7 itemInBin_('I',_2) + 6 itemInBin_('J',_2) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_3:
 - 13 BinUsed_3 + 2 itemInBin_('A',_3) + 5 itemInBin_('B',_3)
 + 7 itemInBin_('C',_3) + 13 itemInBin_('D',_3) + itemInBin_('E',_3)
 + 4 itemInBin_('F',_3) + 9 itemInBin_('G',_3) + 2 itemInBin_('H',_3)
 + 7 itemInBin_('I',_3) + 6 itemInBin_('J',_3) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_4:
 - 13 BinUsed_4 + 2 itemInBin_('A',_4) + 5 itemInBin_('B',_4)
 + 7 itemInBin_('C',_4) + 13 itemInBin_('D',_4) + itemInBin_('E',_4)
 + 4 itemInBin_('F',_4) + 9 itemInBin_('G',_4) + 2 itemInBin_('H',_4)
 + 7 itemInBin_('I',_4) + 6 itemInBin_('J',_4) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_5:
 - 13 BinUsed_5 + 2 itemInBin_('A',_5) + 5 itemInBin_('B',_5)
 + 7 itemInBin_('C',_5) + 13 itemInBin_('D',_5) + itemInBin_('E',_5)
 + 4 itemInBin_('F',_5) + 9 itemInBin_('G',_5) + 2 itemInBin_('H',_5)
 + 7 itemInBin_('I',_5) + 6 itemInBin_('J',_5) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_6:
 - 13 BinUsed_6 + 2 itemInBin_('A',_6) + 5 itemInBin_('B',_6)
 + 7 itemInBin_('C',_6) + 13 itemInBin_('D',_6) + itemInBin_('E',_6)
 + 4 itemInBin_('F',_6) + 9 itemInBin_('G',_6) + 2 itemInBin_('H',_6)
 + 7 itemInBin_('I',_6) + 6 itemInBin_('J',_6) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_7:
 - 13 BinUsed_7 + 2 itemInBin_('A',_7) + 5 itemInBin_('B',_7)
 + 7 itemInBin_('C',_7) + 13 itemInBin_('D',_7) + itemInBin_('E',_7)
 + 4 itemInBin_('F',_7) + 9 itemInBin_('G',_7) + 2 itemInBin_('H',_7)
 + 7 itemInBin_('I',_7) + 6 itemInBin_('J',_7) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_8:
 - 13 BinUsed_8 + 2 itemInBin_('A',_8) + 5 itemInBin_('B',_8)
 + 7 itemInBin_('C',_8) + 13 itemInBin_('D',_8) + itemInBin_('E',_8)
 + 4 itemInBin_('F',_8) + 9 itemInBin_('G',_8) + 2 itemInBin_('H',_8)
 + 7 itemInBin_('I',_8) + 6 itemInBin_('J',_8) <= 0
The_sum_of_all_items_must_be_smaller_than_the_bin_capacity:_bin_9:
 - 13 BinUsed_9 + 2 itemInBin_('A',_9) + 5 itemInBin_('B',_9)
 + 7 itemInBin_('C',_9) + 13 itemInBin_('D',_9) + itemInBin_('E',_9)
 + 4 itemInBin_('F',_9) + 9 itemInBin_('G',_9) + 2 itemInBin_('H',_9)
 + 7 itemInBin_('I',_9) + 6 itemInBin_('J',_9) <= 0
Binaries
BinUsed_0
BinUsed_1
BinUsed_2
BinUsed_3
BinUsed_4
BinUsed_5
BinUsed_6
BinUsed_7
BinUsed_8
BinUsed_9
itemInBin_('A',_0)
itemInBin_('A',_1)
itemInBin_('A',_2)
itemInBin_('A',_3)
itemInBin_('A',_4)
itemInBin_('A',_5)
itemInBin_('A',_6)
itemInBin_('A',_7)
itemInBin_('A',_8)
itemInBin_('A',_9)
itemInBin_('B',_0)
itemInBin_('B',_1)
itemInBin_('B',_2)
itemInBin_('B',_3)
itemInBin_('B',_4)
itemInBin_('B',_5)
itemInBin_('B',_6)
itemInBin_('B',_7)
itemInBin_('B',_8)
itemInBin_('B',_9)
itemInBin_('C',_0)
itemInBin_('C',_1)
itemInBin_('C',_2)
itemInBin_('C',_3)
itemInBin_('C',_4)
itemInBin_('C',_5)
itemInBin_('C',_6)
itemInBin_('C',_7)
itemInBin_('C',_8)
itemInBin_('C',_9)
itemInBin_('D',_0)
itemInBin_('D',_1)
itemInBin_('D',_2)
itemInBin_('D',_3)
itemInBin_('D',_4)
itemInBin_('D',_5)
itemInBin_('D',_6)
itemInBin_('D',_7)
itemInBin_('D',_8)
itemInBin_('D',_9)
itemInBin_('E',_0)
itemInBin_('E',_1)
itemInBin_('E',_2)
itemInBin_('E',_3)
itemInBin_('E',_4)
itemInBin_('E',_5)
itemInBin_('E',_6)
itemInBin_('E',_7)
itemInBin_('E',_8)
itemInBin_('E',_9)
itemInBin_('F',_0)
itemInBin_('F',_1)
itemInBin_('F',_2)
itemInBin_('F',_3)
itemInBin_('F',_4)
itemInBin_('F',_5)
itemInBin_('F',_6)
itemInBin_('F',_7)
itemInBin_('F',_8)
itemInBin_('F',_9)
itemInBin_('G',_0)
itemInBin_('G',_1)
itemInBin_('G',_2)
itemInBin_('G',_3)
itemInBin_('G',_4)
itemInBin_('G',_5)
itemInBin_('G',_6)
itemInBin_('G',_7)
itemInBin_('G',_8)
itemInBin_('G',_9)
itemInBin_('H',_0)
itemInBin_('H',_1)
itemInBin_('H',_2)
itemInBin_('H',_3)
itemInBin_('H',_4)
itemInBin_('H',_5)
itemInBin_('H',_6)
itemInBin_('H',_7)
itemInBin_('H',_8)
itemInBin_('H',_9)
itemInBin_('I',_0)
itemInBin_('I',_1)
itemInBin_('I',_2)
itemInBin_('I',_3)
itemInBin_('I',_4)
itemInBin_('I',_5)
itemInBin_('I',_6)
itemInBin_('I',_7)
itemInBin_('I',_8)
itemInBin_('I',_9)
itemInBin_('J',_0)
itemInBin_('J',_1)
itemInBin_('J',_2)
itemInBin_('J',_3)
itemInBin_('J',_4)
itemInBin_('J',_5)
itemInBin_('J',_6)
itemInBin_('J',_7)
itemInBin_('J',_8)
itemInBin_('J',_9)
End
