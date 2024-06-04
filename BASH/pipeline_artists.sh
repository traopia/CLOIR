#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=experiments/artists_train_eval

source activate artsagenet
cd /home/tliberatore2/Artistic_Influence_prediction
#artists=("martiros-sarian" "maurice-prendergast" "maurice-utrillo" "maxime-maufra" "moise-kisling" "odilon-redon" "pablo-picasso" "paul-cezanne" "paul-gauguin" "paula-modersohn-becker" "pierre-bonnard" "piet-mondrian" "pyotr-konchalovsky" "salvador-dali" "vincent-van-gogh" "anders-zorn" "boris-kustodiev" "camille-corot" "charles-francois-daubigny" "claude-monet" "edouard-manet" "edward-hopper" "eric-fischl" "eugene-boudin" "george-luks" "giovanni-boldini" "gustave-courbet" "henri-fantin-latour" "ilya-repin" "jamie-wyeth" "john-french-sloan" "john-singer-sargent" "nicholas-roerich" "thomas-eakins" "valentin-serov" "vasily-perov" "vasily-polenov" "vasily-surikov" "giovanni-battista-tiepolo" "joshua-reynolds" "thomas-gainsborough" "edward-burne-jones" "ivan-aivazovsky" "john-constable" "orest-kiprensky" "thomas-cole" "william-turner" "peter-paul-rubens" "rembrandt" "alphonse-mucha" "anna-ostroumova-lebedeva" "aubrey-beardsley" "ferdinand-hodler" "ivan-bilibin" "koloman-moser" "raphael-kirchner" "jan-steen" "johannes-vermeer" "ellsworth-kelly" "morris-louis" "sam-francis" "arshile-gorky" "juan-gris" "tarsila-do-amaral" "amedeo-modigliani" "chaim-soutine" "ernst-ludwig-kirchner" "henri-matisse" "childe-hassam" "edgar-degas" "filipp-malyavin" "franz-marc" "henri-de-toulouse-lautrec" "james-mcneill-whistler" "john-henry-twachtman" "mary-cassatt" "pierre-auguste-renoir" "walter-sickert" "willard-metcalf" "william-merritt-chase" "el-greco" "tintoretto" "fernando-botero" "albrecht-durer" "hans-holbein-the-younger" "andy-warhol" "roy-lichtenstein" "ossip-zadkine" "raoul-dufy" "andrea-mantegna" "edvard-munch" "egon-schiele" "frank-auerbach" "georges-braque" "lovis-corinth" "lucian-freud" "marc-chagall" "mark-rothko" "frank-stella" "gene-davis" "helen-frankenthaler" "joan-miro" "max-beckmann" "zinaida-serebriakova" "raphael" "titian" "alfred-sisley" "auguste-rodin" "berthe-morisot" "ilya-mashkov" "jean-metzinger" "wassily-kandinsky" "james-tissot" "mikhail-vrubel" "nikolay-bogdanov-belsky" "theodore-gericault" "theodore-rousseau" "william-james-glackens" "winslow-homer" "giovanni-domenico-tiepolo" "dante-gabriel-rossetti" "eugene-delacroix" "ford-madox-brown" "francisco-goya" "gustave-dore" "gustave-caillebotte" "guy-rose" "parmigianino" "richard-diebenkorn" "willem-de-kooning" "fernand-leger" "adriaen-brouwer" "adriaen-van-ostade" "annibale-carracci" "anthony-van-dyck" "bartolome-esteban-murillo" "caravaggio" "diego-velazquez" "jackson-pollock" "max-pechstein" "frans-hals" "frans-snyders" "gerard-terborch" "gerrit-dou" "guido-reni" "jacob-jordaens" "paul-klee" "m-c-escher" "joaquã\u00adn-sorolla")
artists=("camille-pisarro" "david-burliuk" "georges-seurat" "gustave-loiseau" "henri-edmond-cross" "martiros-sarian" "maurice-prendergast" "pablo-picasso")
#python create_data_loader.py --dataset_name "wikiart" --artist_splits --feature "image_text_features" --feature_extractor_name "all" --num_examples 100 --positive_based_on_similarity
for artist in "${artists[@]}"
do
    # python create_data_loader.py --dataset_name "wikiart" --artist_splits --feature "image_text_features" --feature_extractor_name "$artist" --num_examples 10
    # python Triplet_Network.py --dataset_name "wikiart" --artist_splits --feature "image_text_features" --feature_extractor_name "$artist" --num_examples 10 
    python evaluation.py --dataset_name "wikiart" --artist_splits --feature "image_text_features" --feature_extractor_name "$artist" --num_examples 100 --positive_based_on_similarity
done
conda deactivate
