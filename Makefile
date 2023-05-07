quality:
	black crnn_pom tests
	autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables crnn_pom tests
	isort crnn_pom tests
	pflake8 crnn_pom tests

check:
	black --check crnn_pom tests
	isort --check-only crnn_pom tests
	pflake8 crnn_pom tests