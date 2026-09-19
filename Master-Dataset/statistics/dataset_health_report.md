# Dataset Health Report

**Total images:** 14443  
**Breeds:** 15


## Class Imbalance
- Largest breed: **sahiwal** (3300 images)
- Smallest breed: **kankrej** (600 images)
- Imbalance ratio: **5.50x**
- Mean per breed: 962.9

## Source-Concentration Risk
✅ No breed is overly dependent on a single source dataset.

## Resolution Consistency
⚠ 4 breed(s) skew toward smaller images vs. dataset average (25.7%):

- **ayrshire**: 41.4% small images
- **brown-swiss**: 42.6% small images
- **jaffarabadi**: 49.3% small images
- **nili-ravi**: 46.9% small images

## Grayscale Concentration
Overall grayscale share: **16.17%**

⚠ 1 breed(s) have disproportionately high grayscale share:

- **holstein-friesian**: 40.7% (vs 16.2% average)

## Real vs Synthetic Composition (relevant for 07_split_dataset.py)
| Breed | Real | Synthetic | Real % |
|---|---|---|---|
| sahiwal | 1540 | 1760 | 46.7% |
| tharparkar | 555 | 873 | 38.9% |
| holstein-friesian | 1085 | 0 | 100.0% |
| gir | 993 | 0 | 100.0% |
| red-sindhi | 375 | 551 | 40.5% |
| rathi | 330 | 523 | 38.7% |
| vechur | 318 | 488 | 39.5% |
| jersey | 768 | 0 | 100.0% |
| ayrshire | 631 | 0 | 100.0% |
| umblachery | 287 | 331 | 46.4% |
| brown-swiss | 611 | 0 | 100.0% |
| nili-ravi | 329 | 281 | 53.9% |
| nagori | 359 | 249 | 59.0% |
| jaffarabadi | 606 | 0 | 100.0% |
| kankrej | 503 | 97 | 83.8% |