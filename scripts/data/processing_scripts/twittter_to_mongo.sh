ls -1 ../twitter-files/$jsonfile | while read jsonfile; do mongoimport -d support -c logs --file ../twitter-files/$jsonfile --db db --jsonArray --collection tweets; done
