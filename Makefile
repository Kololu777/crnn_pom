quality:
	black crnn_pom/*
	autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables crnn_pom/*
	isort crnn_pom/*
	flake8 crnn_pom/* --max-line-length 120