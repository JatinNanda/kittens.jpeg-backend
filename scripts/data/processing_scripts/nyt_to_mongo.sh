ls -1 ../nyt-files/archives-json/$jsonfile | while read jsonfile; do mongoimport -d support -c logs --file ../nyt-files/archives-json/$jsonfile --db db --jsonArray --collection nyt; done
