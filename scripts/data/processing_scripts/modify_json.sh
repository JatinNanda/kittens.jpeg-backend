for i in ../nyt-files/archives-json/*.json; do jq '.[0].response.docs' "$i" | sponge "$i" && echo "$i"
done
