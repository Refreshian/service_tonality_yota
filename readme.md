docker exec -it flask-shines_flask_1 bash

docker exec -it flask-shines_flask_1 python train.py

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"data":"все хорошо отлично супер летает"}' \
  http://localhost:5000/tonality_yota/yota_post