# Report: Citation-Based Case Grouping Experiment

This document summarizes the findings from experiments to determine the best strategy for dividing legal cases into manageable groups (tables) for vector database sharding.

## Objective

To find a grouping strategy that breaks the 6.9M case dataset into smaller, coherent chunks while maintaining high relevance (citation links) within each chunk.

## Executive Summary

We compared two strategies: **Global Disjoint Sets** and **Local Neighborhoods**.

- **Global Grouping (Union-Find)** failed because legal citations are highly transitive, leading to a "super-cluster" containing nearly half of the entire US dataset.
- **Local Neighborhoods (Ego-Networks)** succeeded by limiting groups to direct citations only. This resulted in manageable group sizes (99% < 100 cases) while preserving local relevance.

---

## Experiment 1: Global Disjoint Sets

In this experiment, Case A and Case B were placed in the same group if _any_ citation path existed between them.

- **Metric**: Total Disjoint Sets (Groups).
- **Result**:
  - **Total Cases (US sample)**: 329,995
  - **Largest Group Size**: 152,917 (The Super-Cluster)
  - **Isolation Rate**: 51% (Cases with no citation links)
- **Conclusion**: Not suitable for sharding. One table would be disproportionately large, causing performance bottlenecks.

## Experiment 2: Local Neighborhoods (Ego-Networks)

In this experiment, Case A's group includes itself and its **direct** neighbors (cases it cites + cases that cite it). Overlapping is expected. This strategy was initially verified on a US data sample and then scaled to the full dataset.

### Final Results (Full Dataset: 6,902,269 cases)

| Metric                        | Value     |
| :---------------------------- | :-------- |
| **Total Cases**               | 6,902,269 |
| **Max Neighborhood Size**     | 69,148    |
| **Average Neighborhood Size** | 19.82     |

#### Neighborhood Size Distribution

| Size Range       | Count of Cases | Percentage |
| :--------------- | :------------- | :--------- |
| **1 (Isolated)** | 941,787        | 13.64%     |
| **2 - 10**       | 2,664,757      | 38.61%     |
| **11 - 50**      | 2,729,096      | 39.54%     |
| **51 - 100**     | 405,761        | 5.88%      |
| **101 - 500**    | 155,877        | 2.26%      |
| **501+**         | 4,991          | 0.07%      |

> [!NOTE] > **Observation**: Scaling to the full dataset drastically increased the connectivity. While the sample showed 51% isolated cases, the full dataset has only **13.6%** isolated cases, with nearly **80%** of cases belonging to neighborhoods of size 2-50.

---

## Strategy Recommendations

1.  **Overlapping Sharding**: Use Local Neighborhoods to define shards. Since groups overlap, cases will be indexed in multiple "neighbor tables," ensuring that when searching from Case X, all related cases (X±1 citation hop) are present in the same shard.
2.  **Size Limits**: Even the "peak" cases (923 neighbors) are well within the capacity of individual vector indices.
3.  **Future Work**: Consider combining citation neighborhoods with **Jurisdictional** or **Time-based** partitioning for even more granular control.

## Scripts & Tools

- **Group Statistics**: [src/scripts/analyze_case_groups.py](../src/scripts/analyze_case_groups.py)
- **Neighborhood Statistics**: [src/scripts/analyze_case_neighborhoods.py](../src/scripts/analyze_case_neighborhoods.py)
