quality:
	black crnn_pom/* tests
	autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables crnn_pom/* tests
	isort crnn_pom/* tests
	flake8 crnn_pom/* tests --max-line-length 120