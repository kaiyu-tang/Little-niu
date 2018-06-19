import sys


def adjust(s):
    s = list(s)
    s_len = len(s)
    pre, las = 0, s_len - 1
    while pre < las:
        while s[pre] < 'a' and pre < las:
            pre += 1
        while s[las] > 'Z' and pre < las:
            las -= 1
        s[pre], s[las] = s[las], s[pre]
        pre += 1
        las -= 1
    print()
    return ''.join(s[::-1])


while True:
    line = sys.stdin.readline().strip().split()
    if line == '':
        break
    M, N = int(line[0]), int(line[1])
    scores = list(map(int, sys.stdin.readline().strip().split()))
    for i in range(N):
        line = sys.stdin.readline().strip().split()
        if line[0] == 'Q':
            print(max(scores[int(line[1]):int(line[2]) + 1]))
        if line[1] == 'U':
            a, b = int(line[1]), int(line[2])
            scores[a] = b
