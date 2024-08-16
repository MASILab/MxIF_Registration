# Deformable registration of MxIF images using DAPI stains as anchors
This pipeline can be used to perform deformable registration for MxIF images using DAPI stains as anchors. It has 3 main functions:
1. It identifies and masks out areas of tissue deformation and tissue loss by composing the forward and inverse registration fields from the first round of DAPI to the last.
2. It uses the individual rounds of DAPI as anchors and registers the stains associated with each round of DAPI into a unified space.
3. After registering all of the DAPI rounds back to the initial round, it adjusts the intensity differences between the different rounds of DAPI and averages them to make a representative DAPI image that accounts for all rounds.

