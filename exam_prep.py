import numpy as np
from collections.abc import Callable
import re


# def add(x, y):
#     return x + y


# def graph_calc(x: int, y: int, calc_func: Callable[[int, int], int]):
#     return calc_func(x, y)

# result = graph_calc(5, 4, add)
# result_2 = graph_calc(5, 4, lambda x, y: x + y)
# print(result, result_2)


# def sum_of_squares(x:list) -> float:
#     running_sum = 0
#     for i in x:
#         running_sum += i**2
#     return running_sum


# sum_of_sq_short = lambda x: sum([i**2 for i in x])

# print(sum_of_squares([3, 4]))
# print(sum_of_sq_short([3, 4]))

# def weird_sum(x:list) -> float:
#     running_sum = 0
#     i = 0
#     for elem in x:
#         if i % 2 == 0:
#             running_sum += elem**2
#         else:
#             running_sum += elem**3
#         i += 1
#     return running_sum

# weird_sum_short = lambda x: sum([elem**2 if i % 2 == 0 else elem**3 for i, elem in enumerate(x)])

# print(weird_sum([1, 2, 3]))
# print(weird_sum_short([1, 2, 3]))'

# text = """
# Hello John,
# Please contact us at support@example.com for further assistance.
# For career inquiries, reach out to hr@company.org.
# Regards,
# team@example.com"""

# pattern = re.compile(r"^.+ ([a-z]+@[a-z]+\.[a-z]+) .+$", re.MULTILINE)
# print(re.findall(pattern, text))


print(local_math.pi)