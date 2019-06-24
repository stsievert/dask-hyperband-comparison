install:
	sudo yum install -y tmux git htop
	pip install --upgrade dask distributed
	pip install --upgrade git+https://github.com/stsievert/dask-ml@scipy19
	conda install -y pytorch-cpu torchvision-cpu -c pytorch
	pip install skorch --upgrade

worker:
	# Run an c5.4xlarge
	# Scheduler on c5.2xlarge
	export OMP_NUM_THREADS=2; dask-worker --nprocs 5 --nthreads 1 --memory-limit="4GB" 172.31.20.153:8786


# Have twice the number of workers available to use more memory
# 4GB per worker
