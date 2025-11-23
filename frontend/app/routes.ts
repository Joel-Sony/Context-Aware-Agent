import { type RouteConfig, index } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  { path: "start", file: "routes/start.tsx" }
] satisfies RouteConfig;
