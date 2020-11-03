echo 'SSRNet'
python ./SSRNET/CA.py './data/megaage_test.npz' 4 4
echo 'MobileNet'
python ./MobileNet_and_DenseNet/CA_M.py './data/megaage_test.npz' 1
echo 'DenseNet'
python ./MobileNet_and_DenseNet/CA_D.py './data/megaage_test.npz' 4

