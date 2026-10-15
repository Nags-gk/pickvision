import { test } from "node:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { PickVisionStack } from "../lib/pickvision-stack";

function synth() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pkg-"));
  fs.writeFileSync(path.join(dir, "x.py"), "");
  return Template.fromStack(new PickVisionStack(new App(), "T", { packageDir: dir, env: { region: "us-east-2" } }));
}

test("arm64 Python 3.12 Lambda with 2 GB memory", () => {
  synth().hasResourceProperties("AWS::Lambda::Function", { Runtime: "python3.12", Architectures: ["arm64"], MemorySize: 2048 });
});

test("public API is throttled and has an error alarm", () => {
  const t = synth();
  t.hasResourceProperties("AWS::ApiGatewayV2::Stage", { DefaultRouteSettings: { ThrottlingRateLimit: 10, ThrottlingBurstLimit: 20 } });
  t.resourceCountIs("AWS::CloudWatch::Alarm", 1);
  t.resourceCountIs("AWS::ApiGatewayV2::Route", 3);
});
