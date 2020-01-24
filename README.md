## Deepfake Detection Challenge

### Validation and Leaderboard Progress 

| Model (`.scripts/` folder) |Image Size|Epochs|Bag|TTA |Fold|Val     |LB    |Comment                          |
| ---------------|----------|------|---|----|----|--------|------|---------------------------------|
| 2XSPPSeresnext50; 2XTSM Resnet50 | 224       |16,17; 4,5     |1 | None |0  | 0.168, 0.163; 0.158, 0.196  | 0.34033 | Cosine restart at 10; ratio clip; BSize 4; `spp15` single lstm 256 HU `tsm01` TSM with avg consesnus | 
| 2XSPPResnet50; 2XTSM Resnet50 | 224       |6,8; 4,5     |1 | Hflip |0  | 0.183, 0.174; 0.158, 0.196  | 0.34486 | Change clip to ratio clip; `spp14` single lstm 256 HU `tsm01` TSM with avg consesnus | 
| 2XSPPResnet50; 2XTSM Resnet50 | 224       |6,8; 4,5     |1 | Hflip |0  | 0.183, 0.174; 0.158, 0.196  | 0.34683 | `spp14` single lstm 256 HU `tsm01` TSM with avg consesnus | 
| 2XSPPResnet50; 2XTSM Resnet50 | 224       |6,8; 4,5     |1 | NA |0  | 0.183, 0.174; 0.158, 0.196  | 0.35405 | `spp14` single lstm 256 HU `tsm01` TSM with avg consesnus |  
| Resnet50 and one 34; SPPNet|224       |7,8,9     |3 | NA |0  |~0.174 (0.183, 0.182, 0.174)  | 0.362 | `spp14` single lstm 256 HU |  
| Resnet34; SPPNet|224       |7,8,9     |3 | NA |0  |0.1786 (0.179, 0.219, 0.210) | 0.386 | `spp13` single lstm 256 HU |
