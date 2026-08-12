FROM node:24-alpine AS build
WORKDIR /app
ARG VITE_API_URL=http://localhost:8000/api/v1
ARG VITE_MLFLOW_URL=http://localhost:5000
ARG VITE_PREFECT_URL=http://localhost:4200
ENV VITE_API_URL=$VITE_API_URL VITE_MLFLOW_URL=$VITE_MLFLOW_URL VITE_PREFECT_URL=$VITE_PREFECT_URL
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -qO- http://127.0.0.1/health || exit 1

