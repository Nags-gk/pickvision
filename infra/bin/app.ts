#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { PickVisionStack } from "../lib/pickvision-stack";

const app = new App();
new PickVisionStack(app, "PickVision", {
  alertEmail: app.node.tryGetContext("alertEmail") ?? process.env.ALERT_EMAIL,
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION ?? "us-east-2" },
});
