let agents = [];
let foods = [];
let predators = [];

const NUM_AGENTS = 40;
const NUM_FOOD = 6;
const NUM_PREDATORS = 2;

function setup() {
  createCanvas(900, 600);

  for (let i = 0; i < NUM_AGENTS; i++) {
    agents.push(new Agent(random(width), random(height)));
  }

  for (let i = 0; i < NUM_FOOD; i++) {
    foods.push(createVector(random(width), random(height)));
  }

  for (let i = 0; i < NUM_PREDATORS; i++) {
    predators.push(new Predator(random(width), random(height)));
  }
}

function draw() {
  background(10, 14, 18, 40);

  drawField();
  updateFoods();
  updatePredators();

  for (const food of foods) {
    fill(80, 220, 120);
    noStroke();
    circle(food.x, food.y, 12);
  }

  for (const predator of predators) {
    predator.update();
    predator.show();
  }

  for (const agent of agents) {
    agent.update();
    agent.show();
  }

  drawUI();
}

function drawField() {
  noFill();
  stroke(255, 255, 255, 10);
  for (let x = 0; x < width; x += 40) line(x, 0, x, height);
  for (let y = 0; y < height; y += 40) line(0, y, width, y);
}

function updateFoods() {
  for (let i = 0; i < foods.length; i++) {
    if (frameCount % 240 === 0 && random() < 0.25) {
      foods[i] = createVector(random(width), random(height));
    }
  }
}

function updatePredators() {
  if (frameCount % 360 === 0 && random() < 0.5) {
    const idx = floor(random(predators.length));
    predators[idx].target = createVector(random(width), random(height));
  }
}

function drawUI() {
  fill(255);
  noStroke();
  textSize(14);
  text("RDF minimal world", 20, 25);
  text("green: food  red: predator  cyan ring: heat  yellow flash: leap", 20, 45);
}

class Agent {
  constructor(x, y) {
    this.pos = createVector(x, y);
    this.vel = p5.Vector.random2D().mult(random(0.5, 1.5));
    this.acc = createVector(0, 0);

    // RDFっぽい内部変数
    this.M = random(0.90, 0.97);   // 慣性
    this.H = 0;                    // 熱
    this.theta = random(2.0, 3.5); // 跳躍閾値

    // 素流圧への感度
    this.wFood = random(0.6, 1.2);
    this.wPred = random(1.0, 1.8);

    // 内部状態
    this.hunger = random(0.2, 0.8);
    this.fear = 0;
    this.cooldown = 0;
    this.justLeaped = 0;
  }

  update() {
    this.acc.mult(0);

    // 内部状態更新
    this.hunger = constrain(this.hunger + 0.002, 0, 1);
    this.fear *= 0.95;
    if (this.cooldown > 0) this.cooldown--;
    if (this.justLeaped > 0) this.justLeaped--;

    // 現在の期待方向（M·V のラフ版）
    const inertialFlow = this.vel.copy().mult(this.M);

    // 勾配を計算
    const foodForce = this.getFoodForce();
    const predForce = this.getPredatorForce();

    // 合成勾配
    const totalForce = createVector(0, 0);
    totalForce.add(foodForce.copy().mult(this.wFood));
    totalForce.add(predForce.copy().mult(this.wPred));

    // 誤差 = 今の慣性方向と勾配方向のズレ
    const expected = inertialFlow.copy();
    const error = p5.Vector.sub(totalForce, expected).mag();

    // 熱蓄積
    this.H += error * 0.015;
    this.H *= 0.992;

    // 熱が高いほど揺らぎ増加
    const noise = p5.Vector.random2D().mult(this.H * 0.15);
    totalForce.add(noise);

    // 跳躍
    if (this.H > this.theta && this.cooldown === 0) {
      this.leap();
    }

    this.acc.add(totalForce);
    this.vel.add(this.acc);

    // 速度制限
    const maxSpeed = map(this.H, 0, 4, 2.0, 4.0, true);
    this.vel.limit(maxSpeed);

    this.pos.add(this.vel);

    this.wrap();

    // 食べたら hunger を下げる
    for (let i = 0; i < foods.length; i++) {
      if (p5.Vector.dist(this.pos, foods[i]) < 10) {
        this.hunger = max(0, this.hunger - 0.45);
        foods[i] = createVector(random(width), random(height));
      }
    }
  }

  getFoodForce() {
    let nearest = null;
    let bestDist = Infinity;

    for (const food of foods) {
      const d = p5.Vector.dist(this.pos, food);
      if (d < bestDist) {
        bestDist = d;
        nearest = food;
      }
    }

    if (!nearest) return createVector(0, 0);

    const dir = p5.Vector.sub(nearest, this.pos);
    const dist = max(dir.mag(), 1);

    dir.normalize();

    // hunger が高いほど食物引力が強くなる
    const strength = (1 / dist) * (0.3 + this.hunger * 2.5);
    return dir.mult(strength);
  }

  getPredatorForce() {
    let sum = createVector(0, 0);

    for (const predator of predators) {
      const away = p5.Vector.sub(this.pos, predator.pos);
      const dist = away.mag();

      if (dist < 140) {
        away.normalize();

        const strength = map(dist, 0, 140, 2.4, 0, true);
        sum.add(away.mult(strength));
        this.fear = min(1, this.fear + 0.06);
      }
    }

    return sum;
  }

  leap() {
    // 跳躍: 感度の再編
    // ここでは簡略的に food/pred の重みと閾値を少し変える
    this.wFood = constrain(this.wFood + random(-0.35, 0.45), 0.2, 2.0);
    this.wPred = constrain(this.wPred + random(-0.25, 0.40), 0.6, 2.5);
    this.theta = constrain(this.theta + random(-0.25, 0.25), 1.6, 4.0);

    // 進行方向も少し切り替える
    this.vel.rotate(random(-PI / 2, PI / 2));
    this.vel.mult(random(0.8, 1.2));

    this.H *= 0.25; // 熱解放
    this.cooldown = 60;
    this.justLeaped = 20;
  }

  wrap() {
    if (this.pos.x < 0) this.pos.x = width;
    if (this.pos.x > width) this.pos.x = 0;
    if (this.pos.y < 0) this.pos.y = height;
    if (this.pos.y > height) this.pos.y = 0;
  }

  show() {
    push();
    translate(this.pos.x, this.pos.y);

    // 熱の可視化
    noFill();
    stroke(80, 220, 255, map(this.H, 0, 4, 20, 180, true));
    circle(0, 0, 10 + this.H * 10);

    // 跳躍フラッシュ
    if (this.justLeaped > 0) {
      noFill();
      stroke(255, 220, 80, 180);
      circle(0, 0, 22);
    }

    // 本体
    noStroke();
    const bodyColor = lerpColor(
      color(120, 180, 255),
      color(255, 120, 120),
      this.fear
    );
    fill(bodyColor);

    rotate(this.vel.heading());
    triangle(8, 0, -6, -5, -6, 5);

    pop();
  }
}

class Predator {
  constructor(x, y) {
    this.pos = createVector(x, y);
    this.vel = p5.Vector.random2D().mult(1.2);
    this.target = createVector(random(width), random(height));
  }

  update() {
    // 近いエージェントを追う
    let nearest = null;
    let bestDist = Infinity;

    for (const agent of agents) {
      const d = p5.Vector.dist(this.pos, agent.pos);
      if (d < bestDist) {
        bestDist = d;
        nearest = agent;
      }
    }

    let desired;
    if (nearest && bestDist < 180) {
      desired = p5.Vector.sub(nearest.pos, this.pos);
    } else {
      desired = p5.Vector.sub(this.target, this.pos);
      if (desired.mag() < 20) {
        this.target = createVector(random(width), random(height));
        desired = p5.Vector.sub(this.target, this.pos);
      }
    }

    desired.normalize().mult(0.18);
    this.vel.add(desired);
    this.vel.limit(2.4);
    this.pos.add(this.vel);

    if (this.pos.x < 0) this.pos.x = width;
    if (this.pos.x > width) this.pos.x = 0;
    if (this.pos.y < 0) this.pos.y = height;
    if (this.pos.y > height) this.pos.y = 0;
  }

  show() {
    push();
    translate(this.pos.x, this.pos.y);
    rotate(this.vel.heading());

    noStroke();
    fill(255, 80, 80);
    triangle(12, 0, -8, -6, -8, 6);

    pop();
  }
}