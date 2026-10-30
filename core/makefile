all: prepare verify view

clean:
	rm -rf ../data/csv/
	rm -rf ../data/onnx/
	rm -rf ../data/vnnlib/
	rm -rf ../data/log/

train:
	python3 main.py --trainer=semcom --n_latent=32
	python3 main.py --trainer=semcom --n_latent=64
	python3 main.py --trainer=pgm --n_latent=32
	python3 main.py --trainer=pgm --n_latent=64

prepare:
	python3 main.py --scenario=exp_prepare --experiment=fashion_mnist --n_sample_per_class=100
	python3 main.py --scenario=prepare_e2e_csv --experiment=fashion_mnist --n_sample_per_class=100
	python3 main.py --scenario=prepare_split_csv --experiment=fashion_mnist --n_sample_per_class=100
	python3 main.py --scenario=exp_prepare --experiment=cifar10 --n_sample_per_class=50
	python3 main.py --scenario=prepare_e2e_csv --experiment=cifar10 --n_sample_per_class=50
	python3 main.py --scenario=prepare_split_csv --experiment=cifar10 --n_sample_per_class=50

verify:
	python3 main.py --scenario=verify --experiment=cifar10 --verifier=e2e
	python3 main.py --scenario=verify --experiment=cifar10 --verifier=e2e_mip
	python3 main.py --scenario=verify --experiment=cifar10 --verifier=abcrown

attack:
	python3 main.py --scenario=attack --verifier=pgm --experiment=fashion_mnist
	python3 main.py --scenario=attack --verifier=pgm --experiment=cifar10
view:
	python3 main.py --scenario=compare
	python3 main.py --scenario=compare_split

check_cex:
	python3 main.py --scenario=check_cex --verifier=e2e --experiment=fashion_mnist
	python3 main.py --scenario=check_cex --verifier=e2e --experiment=cifar10

plot:
	python3 main.py --scenario=plot_cactus
	python3 main.py --scenario=plot_compare
	python3 main.py --scenario=plot_compare_number_of_branch
	python3 main.py --scenario=plot_compare_iteration
	python3 main.py --scenario=plot_ablation

verify_to_export_cex:
	python3 main.py --scenario=verify_to_export_cex --verifier=e2e_mip --experiment=fashion_mnist --verbose
	python3 main.py --scenario=verify_to_export_cex --verifier=e2e_mip --experiment=cifar10 --verbose

train_ablation_fashion_mnist:
	python3 main.py --trainer=pgm --max_epoch=50 --experiment=fashion_mnist --adv_snr_threshold=0
	python3 main.py --trainer=pgm --max_epoch=50 --experiment=fashion_mnist --adv_snr_threshold=-5
	python3 main.py --trainer=pgm --max_epoch=50 --experiment=fashion_mnist --adv_snr_threshold=-10

train_ablation_cifar10:
	python3 main.py --trainer=pgm --device=cuda:1 --max_epoch=100 --experiment=cifar10 --dataset=torch/cifar10 --adv_snr_threshold=0
	python3 main.py --trainer=pgm --device=cuda:1 --max_epoch=100 --experiment=cifar10 --dataset=torch/cifar10 --adv_snr_threshold=-5
	python3 main.py --trainer=pgm --device=cuda:1 --max_epoch=100 --experiment=cifar10 --dataset=torch/cifar10 --adv_snr_threshold=-10

prepare_ablation_fashion_mnist:
	python3 main.py --scenario=exp_prepare --experiment=ablation_fashion_mnist --n_sample_per_class=10

prepare_ablation_cifar10:
	python3 main.py --scenario=exp_prepare --experiment=ablation_cifar10 --n_sample_per_class=10

verify_ablation:
	python3 main.py --scenario=verify --experiment=ablation_fashion_mnist --verifier=e2e_mip
	python3 main.py --scenario=verify --experiment=ablation_fashion_mnist --verifier=e2e
	python3 main.py --scenario=verify --experiment=ablation_fashion_mnist --verifier=abcrown

proof:
	cd ../lean && lake build && lake env lean Check.lean
