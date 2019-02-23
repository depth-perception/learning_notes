# leetcode刷题

## 1.**Two Sum**

Given an array of integers, return **indices** of the two numbers such that they add up to a specific target.

You may assume that each input would have **exactly** one solution, and you may not use the *same* element twice.

**Example:**

```
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

```

```python
class Solution(object):
    def twoSum(self, nums, target):
    
        for index_1,value_1 in enumerate(nums):
            for index_2,value_2 in enumerate(nums):
                if index_1==index_2:
                    continue
                elif value_1+value_2==target:
                    return [index_1,index_2]

```

## 2.Two Sum II - Input array is sorted    

Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

**Note:**

- Your returned answers (both index1 and index2) are not zero-based.
- You may assume that each input would have *exactly* one solution and you may not use the *same* element twice.

**Example:**

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
```

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        for index_1,value_1 in enumerate(numbers):
            for index_2,value_2 in enumerate(numbers):
                if index_1>=index_2:
                    continue
                elif value_1+value_2==target :
                    return [index_1+1,index_2+1]
```

## 3.3Sum

Given an array `nums` of *n* integers, are there elements *a*, *b*, *c* in `nums` such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:**

The solution set must not contain duplicate triplets.

**Example:**

```
Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums)<3:
            return []
        nums.sort()
        s=set()
        for i,j in enumerate(nums):
            k,r=i+1,len(nums)-1
            while k<r:
                sum=j+nums[k]+nums[r]
                if sum==0:
                    s.add((j,nums[k],nums[r]))
                    k+=1
                    r-=1
                elif sum>0:
                    r-=1
                elif sum<0:
                    k+=1
        return list(s)
                
        
```

