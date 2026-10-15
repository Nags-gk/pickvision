import * as path from "path";
import { CfnOutput, Duration, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as apigw from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import * as cw from "aws-cdk-lib/aws-cloudwatch";
import * as cwActions from "aws-cdk-lib/aws-cloudwatch-actions";
import * as sns from "aws-cdk-lib/aws-sns";
import * as subs from "aws-cdk-lib/aws-sns-subscriptions";

export interface PickVisionProps extends StackProps {
  /** directory holding the prebuilt Lambda package (see scripts/build_lambda.sh) */
  packageDir?: string;
  alertEmail?: string;
  allowedOrigins?: string[];
}

/** Serverless inference API for PickNet: API Gateway HTTP API -> Lambda (ONNX Runtime, arm64). */
export class PickVisionStack extends Stack {
  constructor(scope: Construct, id: string, props: PickVisionProps = {}) {
    super(scope, id, props);

    const logGroup = new logs.LogGroup(this, "InferenceLogs", { retention: logs.RetentionDays.TWO_WEEKS });
    const fn = new lambda.Function(this, "Inference", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: "pickvision.lambda_handler.handler",
      code: lambda.Code.fromAsset(props.packageDir ?? path.join(__dirname, "..", "..", "..", "build", "lambda")),
      memorySize: 2048,
      timeout: Duration.seconds(30),
      logGroup,
      environment: { OMP_NUM_THREADS: "2" },
      description: "PickNet suction-grasp inference (ONNX Runtime)",
    });

    const api = new apigw.HttpApi(this, "Api", {
      description: "PickVision inference API",
      corsPreflight: {
        allowOrigins: props.allowedOrigins ?? ["https://nags-gk.github.io", "http://localhost:8000"],
        allowMethods: [apigw.CorsHttpMethod.GET, apigw.CorsHttpMethod.POST],
        allowHeaders: ["content-type"],
      },
    });
    const integ = new HttpLambdaIntegration("InferenceInt", fn);
    api.addRoutes({ path: "/v1/picks", methods: [apigw.HttpMethod.POST], integration: integ });
    api.addRoutes({ path: "/healthz", methods: [apigw.HttpMethod.GET], integration: integ });
    api.addRoutes({ path: "/metrics", methods: [apigw.HttpMethod.GET], integration: integ });
    // public, unauthenticated endpoint: keep abuse and cost bounded
    const stage = api.defaultStage!.node.defaultChild as apigw.CfnStage;
    stage.defaultRouteSettings = { throttlingRateLimit: 10, throttlingBurstLimit: 20 };

    const topic = new sns.Topic(this, "Alerts", { displayName: "PickVision alerts" });
    if (props.alertEmail) topic.addSubscription(new subs.EmailSubscription(props.alertEmail));
    new cw.Alarm(this, "InferenceErrors", {
      alarmName: "pickvision-INFERENCE_ERRORS",
      metric: fn.metricErrors({ period: Duration.minutes(5) }),
      threshold: 1, evaluationPeriods: 1,
      comparisonOperator: cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cw.TreatMissingData.NOT_BREACHING,
    }).addAlarmAction(new cwActions.SnsAction(topic));

    new CfnOutput(this, "ApiUrl", { value: api.apiEndpoint });
  }
}
