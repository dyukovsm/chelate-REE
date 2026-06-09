#!/bin/bash

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

# Al Fe Cr Nd U Y La Sm Gd Dy Tm Hf 

destination_folder="000000_analysis/Element_name"
shopt -s nullglob

for dir_name in "${dirs[@]}"; do
    if [ -d "$dir_name" ]; then # Check if the directory exists
        echo "Processing directory: $dir_name"
        files_to_copy="$dir_name/PRO_NPT_PARRINELLO_*.xvg"
        matching_files=( $files_to_copy )
        if [ ${#matching_files[@]} -gt 0 ]; then
            cp -v "${matching_files[@]}" "$destination_folder/"
        else
            echo "  No files matching 'PRO_NPT_PARRINELLO_*.xvg' found in $dir_name. Skipping copy for this directory."
        fi
    else
        echo "Directory not found: $dir_name. Skipping."
    fi
done

shopt -u nullglob
