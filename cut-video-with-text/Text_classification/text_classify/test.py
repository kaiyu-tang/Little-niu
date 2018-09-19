#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/8/20 下午3:28  
# @Author  : Kaiyu  
# @Site    :   
# @File    : test.py
import sys

if __name__ == "__main__":
    n = int(input())
    count = 0
    res = 0
    ma = 0
    if n < 2:
        print("0 0")
    else:
        nums = list(map(int, input().split()))
        flag = True
        buy = nums[0]
        for i in range(1, len(nums)):
            if flag and nums[i] < buy:
                buy = nums[i]
            elif flag == False and nums[i] < nums[i - 1]:
                count += 2
                flag = True
                buy = nums[i]
                res += ma
                ma = 0
            else:
                flag = False
                if ma < nums[i] - buy:
                    ma = nums[i] - buy
        if ma:
            res += ma
            count += 2
        print("{} {}".format(res, count))
