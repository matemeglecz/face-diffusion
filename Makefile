host_ssh_port=
image_name=megleczm/celeba_diffusion
container_name=
data_path=/mnt/hdd2/datasets/

build:
	docker build --tag $(image_name) .

push:
	docker push $(image_name) 

pull:
	docker pull $(image_name) 

stop:
	docker stop $(container_name)

ssh: build
	nvidia-docker run \
	-dt \
	--shm-size 16G \
	-p $(host_ssh_port):22 \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	--name $(container_name) \
	-v $(shell pwd):/workspace \
	-v $(data_path):/data \
	$(image_name) \
	/usr/sbin/sshd -D

