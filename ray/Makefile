HOST := ssievert@opt-a002.discovery.wisc.edu

up:
	scp Makefile *.yaml *.py $(HOST):~/ray

down:
	scp $(HOST):~/ray/out.zip .

clean:
	rm model-histories/*
	rm out/*
