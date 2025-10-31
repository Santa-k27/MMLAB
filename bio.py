#Global Alignment

import seaborn as sns
import matplotlib.pyplot as plt

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Needleman-Wunsch global alignment.
    Returns (aligned_seq1, aligned_seq2, score)
    """
    n, m = len(seq1), len(seq2)

    # ---- 1. Build scoring matrix ---------------------------------
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            left = dp[i][j-1] + gap
            up   = dp[i-1][j] + gap
            dp[i][j] = max(diag, left, up)

    # ---- 2. Traceback ---------------------------------------------
    align1, align2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))

    return align1, align2, dp[n][m], dp   # dp is returned for printing


# ------------------------------------------------------------------
def print_matrix(seq1, seq2, dp, gap=-1):
    """
    Print the DP matrix with seq1 on the *columns* and seq2 on the *rows*.
    """
    n, m = len(seq1), len(seq2)

    # Header (seq1 on top)
    header = "   " + " ".join(f"{c:>3}" for c in " " + seq1)
    print(header)

    # Rows (seq2 on the left)
    for i in range(n + 1):
        row_label = seq2[i-1] if i > 0 else " "
        row = f"{row_label} " + " ".join(f"{dp[i][j]:>3}" for j in range(m + 1))
        print(row)


seq1 = "TGGTG"      # <-- will be shown on *columns*
seq2 = "ATCGT"   # <-- will be shown on *rows*

a1, a2, score, dp = needleman_wunsch(seq1, seq2)

print("=== Needleman-Wunsch (seq1 = columns) ===")
print(f"Seq1 (cols): {seq1}")
print(f"Seq2 (rows): {seq2}\n")
print(f"Alignment score: {score}")
print(f"   {a1}")
print(f"   {a2}\n")

print("Scoring matrix:")
print_matrix(seq1,seq2,dp)

def plot_traceback(seq1, seq2, dp):
    n, m = len(seq1), len(seq2)
    path_i, path_j = [n], [m]

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (1 if seq1[i-1] == seq2[j-1] else -1):
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] - 2:
            i -= 1
        else:
            j -= 1
        path_i.append(i)
        path_j.append(j)

    plt.figure(figsize=(8, 6))
    sns.heatmap(dp, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=["-"] + list(seq1),
                yticklabels=["-"] + list(seq2))
    plt.plot([j + 0.5 for j in path_j[::-1]], [i + 0.5 for i in path_i[::-1]],
             color="red", linewidth=2, marker="o")
    plt.title("Traceback Path (Optimal Alignment)")
    plt.xlabel("Sequence 1")
    plt.ylabel("Sequence 2")
    plt.show()

plot_traceback(seq1, seq2, dp)

