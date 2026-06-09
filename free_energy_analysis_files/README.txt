put .sh file inside workspace and run it. it should make a new directory 000000_analysis/Element_name
ensure that the element Element_name inside the .sh on line: destination_folder="000000_analysis/Element_name" 
is replaced with the name of the element you're trying to extract the .xvg files for. Also ensure that the 
folders specified at the top of the file, are the same as the statepoint id that correspond to the element name,
the information on statepoint id can be found in legend.txt. Meaning if you're trying to extract data for Yttrium (Y)
change destination_folder="000000_analysis/Element_name" -> destination_folder="000000_analysis/Y" and ensure that the
folder IDs match the statepoints:

declare -a dirs=(
"8704a7a5fccb0469221c9c71d42d4d74"
"c2378ad01c4a6ef356874297fe3ee641"
"b7209bcae548ce9461680872afa4651e"
"26b0362c9659fa03487896b89b542fb8"
"22afc67e89000082423034aac5257049"
"52a7e7af949a595ce7aaa0237fab35b0"
"0e0c6307d4b50bb54b070fe7daee9a5d"
"53c2c3818dfcd40b50358556d745e120"
"60aab5212c27a4654832336ad2d02e2f"
"4456577a353f5f87bb4b1634146c6617"
"cf9d1762afe8892be6e638b23d806ff3"
"a6a84dc635503ca1211921de532bfce0"
"f053dc585dc71db31dc5b55634dd9c14"
"5852d76af53a48063b725c0512fbecac"
"46e0f20558d5a32723178af0deaf59bf"
"b950541c947113da4a884b7f77342260"
"eae208aebd0a5564cd76c6a9df0b0e2b"
"b75d45a6423bfb17855d2cdfc1d3533a"
"4807fccc611a9ba9d1352d7cadec1993"
"64bfdc360d5f4cee65f46d38fa1f752a"
"95e5279b414f8a52882b1bd0d1436ec6"
"f9f52a929ff962ff2f38a472395c87af"
"107cde6131f2c142c23ae65742a1da47"
"67939a4b2800e7f6b9c6c62a547fa6f7"
"31a6775f9411e640beb7b6c9b4dc8725"
"da3838bb28d008d9ea7a1e9827c90af4"
"48a2ebf2d1b7d709823da847a546a5dd"
"58ef661851e79d82d9349cc77864030a"
"dae05e09fdd294d5778b15536695ed60"
)

where all the IDs come from legend.txt lines:

 8704a7a5fccb0469221c9c71d42d4d74                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.0}
 c2378ad01c4a6ef356874297fe3ee641                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.2}
 b7209bcae548ce9461680872afa4651e                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.4}
 26b0362c9659fa03487896b89b542fb8                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.6}
 22afc67e89000082423034aac5257049                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.8}
 52a7e7af949a595ce7aaa0237fab35b0                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 0.9}
 0e0c6307d4b50bb54b070fe7daee9a5d                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.0, 'lambda_ELE': 1.0}
 53c2c3818dfcd40b50358556d745e120                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.05, 'lambda_ELE': 1.0}
 60aab5212c27a4654832336ad2d02e2f                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.1, 'lambda_ELE': 1.0}
 4456577a353f5f87bb4b1634146c6617                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.15, 'lambda_ELE': 1.0}
 cf9d1762afe8892be6e638b23d806ff3                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.2, 'lambda_ELE': 1.0}
 a6a84dc635503ca1211921de532bfce0                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.25, 'lambda_ELE': 1.0}
 f053dc585dc71db31dc5b55634dd9c14                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.3, 'lambda_ELE': 1.0}
 5852d76af53a48063b725c0512fbecac                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.35, 'lambda_ELE': 1.0}
 46e0f20558d5a32723178af0deaf59bf                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.4, 'lambda_ELE': 1.0}
 b950541c947113da4a884b7f77342260                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.425, 'lambda_ELE': 1.0}
 eae208aebd0a5564cd76c6a9df0b0e2b                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.45, 'lambda_ELE': 1.0}
 b75d45a6423bfb17855d2cdfc1d3533a                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.475, 'lambda_ELE': 1.0}
 4807fccc611a9ba9d1352d7cadec1993                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.5, 'lambda_ELE': 1.0}
 64bfdc360d5f4cee65f46d38fa1f752a                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.55, 'lambda_ELE': 1.0}
 95e5279b414f8a52882b1bd0d1436ec6                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.6, 'lambda_ELE': 1.0}
 f9f52a929ff962ff2f38a472395c87af                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.65, 'lambda_ELE': 1.0}
 107cde6131f2c142c23ae65742a1da47                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.7, 'lambda_ELE': 1.0}
 67939a4b2800e7f6b9c6c62a547fa6f7                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.75, 'lambda_ELE': 1.0}
 31a6775f9411e640beb7b6c9b4dc8725                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.8, 'lambda_ELE': 1.0}
 da3838bb28d008d9ea7a1e9827c90af4                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.85, 'lambda_ELE': 1.0}
 48a2ebf2d1b7d709823da847a546a5dd                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.9, 'lambda_ELE': 1.0}
 58ef661851e79d82d9349cc77864030a                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 0.95, 'lambda_ELE': 1.0}
 dae05e09fdd294d5778b15536695ed60                {'metal': 'Y', 'replicate': 0, 'lambda_LJ': 1.0, 'lambda_ELE': 1.0}
