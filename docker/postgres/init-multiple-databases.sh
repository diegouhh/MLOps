#!/bin/sh
set -eu

for database in mlflow prefect; do
  exists=$(psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --tuples-only --command "SELECT 1 FROM pg_database WHERE datname = '$database'" | tr -d '[:space:]')
  if [ "$exists" != "1" ]; then
    psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --command "CREATE DATABASE $database"
  fi
done

