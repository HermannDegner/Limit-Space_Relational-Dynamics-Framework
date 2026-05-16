"""
RDL生態系シミュレーション v2
修正点:
- 食料獲得量増加・移動コスト減少
- 人間に eat_grass 追加
- 狩猟ターゲット追跡改善
- hunger閾値・増加速度調整
"""

import random
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(42); np.random.seed(42)

TERRAIN_PLAIN, TERRAIN_WATER, TERRAIN_ROCK, TERRAIN_FOREST = 0,1,2,3


class Ecosystem:
    def __init__(self, size=40):
        self.size = size
        self.t = 0
        self.terrain = self._gen_terrain()
        self.grass   = {}
        self.berries = {}
        self.fish    = {}
        self._init_resources()

    def _gen_terrain(self):
        s = self.size
        g = [[TERRAIN_PLAIN]*s for _ in range(s)]
        # 川
        ry = s // 3
        for x in range(s):
            for dy in range(-1, 2):
                y = ry + dy + random.randint(-1,1)
                if 0 <= y < s: g[y][x] = TERRAIN_WATER
        # 池x2
        for cx,cy in [(8,28),(30,10)]:
            for dx in range(-3,4):
                for dy in range(-3,4):
                    if dx*dx+dy*dy<=9:
                        nx,ny=cx+dx,cy+dy
                        if 0<=nx<s and 0<=ny<s: g[ny][nx]=TERRAIN_WATER
        # 岩x5
        for _ in range(5):
            cx,cy=random.randint(3,s-3),random.randint(3,s-3)
            for dx in range(-2,3):
                for dy in range(-2,3):
                    if abs(dx)+abs(dy)<=2:
                        nx,ny=cx+dx,cy+dy
                        if 0<=nx<s and 0<=ny<s and g[ny][nx]==TERRAIN_PLAIN:
                            g[ny][nx]=TERRAIN_ROCK
        # 森x5
        for _ in range(5):
            cx,cy=random.randint(4,s-4),random.randint(4,s-4)
            for dx in range(-4,5):
                for dy in range(-4,5):
                    if dx*dx+dy*dy<=14:
                        nx,ny=cx+dx,cy+dy
                        if 0<=nx<s and 0<=ny<s and g[ny][nx]==TERRAIN_PLAIN:
                            g[ny][nx]=TERRAIN_FOREST
        return g

    def _init_resources(self):
        s=self.size
        for y in range(s):
            for x in range(s):
                t=self.terrain[y][x]
                if t==TERRAIN_PLAIN:
                    self.grass[(x,y)]=random.uniform(0.4,0.9)
                elif t==TERRAIN_FOREST:
                    self.grass[(x,y)]=random.uniform(0.3,0.7)
                    if random.random()<0.35:
                        self.berries[(x,y)]={"abundance":random.uniform(0.4,0.9),"regen":random.uniform(0.006,0.015)}
                elif t==TERRAIN_WATER:
                    if random.random()<0.5:
                        self.fish[(x,y)]={"abundance":random.uniform(0.4,0.9),"regen":random.uniform(0.008,0.02)}

    def step(self):
        for p,v in self.grass.items(): self.grass[p]=min(1.0,v+0.01*(1-v))
        for p,v in self.berries.items(): v["abundance"]=min(1.0,v["abundance"]+v["regen"]*(1-v["abundance"]))
        for p,v in self.fish.items(): v["abundance"]=min(1.0,v["abundance"]+v["regen"]*(1-v["abundance"]))
        self.t+=1

    def is_passable(self,x,y):
        if not(0<=x<self.size and 0<=y<self.size): return False
        return self.terrain[y][x]!=TERRAIN_ROCK

    def terrain_at(self,x,y):
        if 0<=x<self.size and 0<=y<self.size: return self.terrain[y][x]
        return TERRAIN_ROCK

    def eat_grass(self,pos):
        if pos in self.grass and self.grass[pos]>0.05:
            amt=min(self.grass[pos],random.uniform(0.15,0.35))
            self.grass[pos]=max(0,self.grass[pos]-amt)
            return True,amt*55
        return False,0

    def eat_berry(self,pos):
        if pos in self.berries and self.berries[pos]["abundance"]>0.05:
            b=self.berries[pos]
            food=random.uniform(20,40)*(0.5+b["abundance"]/2)
            b["abundance"]=max(0,b["abundance"]-random.uniform(0.1,0.25))
            return True,food
        return False,0

    def eat_fish(self,pos):
        if pos in self.fish and self.fish[pos]["abundance"]>0.05:
            f=self.fish[pos]
            food=random.uniform(22,42)*(0.5+f["abundance"]/2)
            f["abundance"]=max(0,f["abundance"]-random.uniform(0.08,0.2))
            return True,food
        return False,0

    def nearest_resource(self,pos,rd,k=4):
        nodes=list(rd.keys())
        nodes.sort(key=lambda p:abs(p[0]-pos[0])+abs(p[1]-pos[1]))
        return [n for n in nodes[:k] if rd[n] if isinstance(rd[n],float) and rd[n]>0.05
                or isinstance(rd[n],dict) and rd[n].get("abundance",0)>0.05]

    def nearest_passable_resource(self,pos,rd,k=4):
        nodes=list(rd.keys())
        nodes.sort(key=lambda p:abs(p[0]-pos[0])+abs(p[1]-pos[1]))
        result=[]
        for n in nodes:
            v=rd[n]
            abund=v if isinstance(v,(float,int)) else v.get("abundance",0)
            if abund>0.05: result.append(n)
            if len(result)>=k: break
        return result

    def resource_summary(self):
        return sum(self.grass.values()), sum(v["abundance"] for v in self.berries.values()), sum(v["abundance"] for v in self.fish.values())


class SSDAgent:
    def __init__(self,name,preset,env,roster,start_pos):
        self.name=name
        self.x,self.y=start_pos
        self.env=env
        self.roster=roster
        self.alive=True
        self.age=0

        self.hunger =random.uniform(15,30)
        self.fatigue=random.uniform(10,30)
        self.injury =0.0
        self.fear   =0.0

        self.state      ="Idle"
        self.action_type=None
        self.target_pos =None
        self.coop_target=None

        p=preset
        self.species   =p["species"]
        self.size_class=p.get("size_class","medium")
        self.dopamine      =p.get("dopamine",     0.5)
        self.serotonin     =p.get("serotonin",    0.5)
        self.oxytocin      =p.get("oxytocin",     0.5)
        self.noradrenaline =p.get("noradrenaline",0.5)
        self.risk_tolerance=p.get("risk_tolerance",0.5)
        self.stamina       =p.get("stamina",      0.6)

        self.kappa=defaultdict(lambda:0.05)
        for act,val in p.get("kappa_init",{}).items():
            self.kappa[act]=val
        self.kappa_min=0.05
        self.G0=0.1; self.g=0.05
        self.eta   =0.12*(1+self.dopamine)
        self.E     =0.0
        self.alpha =0.5*(1.2-self.serotonin)
        self.beta_E=0.15
        self.lambda_forget=0.012
        self.lambda_forget_other=0.002
        self.rho=0.03
        self.Theta0=1.0; self.a1=0.5; self.a2=0.4
        self.h0=0.18; self.gamma=0.8
        self.T=0.3; self.T0=0.3; self.c1=0.7; self.c2=0.6

        # hunger閾値: speciesで調整
        self.TH_H=45.0; self.TH_F=60.0; self.TH_I=30.0

        self.rel=defaultdict(float)
        self.log=[]

    def pos(self): return(self.x,self.y)
    def dist_to(self,p): return abs(self.x-p[0])+abs(self.y-p[1])

    def move_towards(self,target,avoid_water=False):
        tx,ty=target
        cands=[]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=self.x+dx,self.y+dy
            if self.env.is_passable(nx,ny):
                if avoid_water and self.env.terrain_at(nx,ny)==TERRAIN_WATER: continue
                cands.append((abs(nx-tx)+abs(ny-ty),nx,ny))
        if cands:
            cands.sort()
            if random.random()<self.T*0.25 and len(cands)>1:
                _,nx,ny=random.choice(cands[:2])
            else:
                _,nx,ny=cands[0]
            self.x,self.y=nx,ny
        self.fatigue=min(120,self.fatigue+0.25)
        return(self.x,self.y)==target

    def alignment_flow(self,a,mp): return(self.G0+self.g*self.kappa[a])*mp
    def update_kappa(self,a,success,reward,chosen):
        for at in list(self.kappa.keys()):
            if at!=chosen: self.kappa[at]=max(self.kappa_min,self.kappa[at]-self.lambda_forget_other)
        k=self.kappa[a]
        w=self.eta*reward if success else -self.rho*(k**2)
        d=self.lambda_forget*(k-self.kappa_min)
        self.kappa[a]=max(self.kappa_min,k+w-d)
    def update_heat(self,mp,proc):
        self.E+=self.alpha*max(0,mp-proc)-self.beta_E*self.E
        self.E=max(0,self.E)
    def update_temperature(self):
        vals=list(self.kappa.values())
        ent=np.std(vals) if len(vals)>1 else 0.5
        self.T=max(0.1,min(1.0,self.T0+self.c1*self.E-self.c2*ent))
    def check_leap(self):
        mk=np.mean(list(self.kappa.values())) if self.kappa else 0.1
        Theta=self.Theta0+self.a1*mk-self.a2*(self.fatigue/100)
        h=self.h0*np.exp((self.E-Theta)/self.gamma)
        return random.random()<1-np.exp(-h),h,Theta

    def nearby(self,radius=5,species_filter=None):
        return[a for n,a in self.roster.items()
               if n!=self.name and a.alive
               and(species_filter is None or a.species in species_filter)
               and self.dist_to(a.pos())<=radius]

    def _log(self,t,action,extra=None):
        e={"t":t,"name":self.name,"species":self.species,"action":action,
           "hunger":round(self.hunger,1),"fatigue":round(self.fatigue,1),
           "injury":round(self.injury,1),"fear":round(self.fear,1),
           "E":round(self.E,2),"T":round(self.T,2)}
        if extra: e.update(extra)
        self.log.append(e)

    def step(self,t):
        if not self.alive: return
        self.age+=1
        sp={"small":0.9,"medium":1.2,"large":1.7}.get(self.size_class,1.5)
        self.hunger =min(120,self.hunger +sp)
        self.fatigue=min(120,self.fatigue+0.6)
        self.injury =min(120,self.injury +0.01*self.fatigue/100)
        self.fear   =max(0,  self.fear   -2.5)

        # ノルアドレナリン緊急モード（草食動物）
        if self.species=="herbivore":
            preds=self.nearby(radius=6,species_filter=["predator"])
            humans=self.nearby(radius=4,species_filter=["human"])
            threats=preds+humans
            if threats:
                self.fear=min(100,self.fear+18*self.noradrenaline)
                if self.fear>25:
                    self._flee(threats[0],t); return

        # 死亡判定
        if self.hunger>=100 or self.injury>=100 or self.fatigue>=120:
            leap,h,Theta=self.check_leap()
            if leap or self.hunger>=112 or self.injury>=110 or self.fatigue>=120:
                self.alive=False
                self._log(t,"death",{"h":round(h,3),"Theta":round(Theta,2)}); return

        self.update_temperature()

        # 移動継続
        if self.state=="Moving" and self.target_pos:
            aw=(self.species in("herbivore",))
            # 狩猟ターゲット追跡
            if self.action_type in("hunt","hunt_herb") and self.coop_target and self.coop_target.alive:
                self.target_pos=self.coop_target.pos()
            arrived=self.move_towards(self.target_pos,avoid_water=aw)
            self._log(t,f"move_{self.action_type}")
            if arrived: self.state="Arrived"
            return

        if self.state=="Arrived":
            self._execute(t)
            self.state="Idle"; self.action_type=None; self.target_pos=None
            return

        if self.state=="Idle":
            self._decide(t)

    def _flee(self,threat,t):
        tx,ty=threat.pos()
        dx=self.x-tx; dy=self.y-ty
        opts=[]
        for ddx,ddy in[(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=self.x+ddx,self.y+ddy
            if self.env.is_passable(nx,ny) and self.env.terrain_at(nx,ny)!=TERRAIN_WATER:
                score=(nx-tx)**2+(ny-ty)**2
                opts.append((-score,nx,ny))
        if opts:
            opts.sort(); _,nx,ny=opts[0]
            self.x,self.y=nx,ny
        self.fatigue=min(120,self.fatigue+1.2)
        self.update_heat(self.fear/100,0)
        self._log(t,"flee")

    def _decide(self,t):
        pr=self._pressures()
        ut=self._utilities(pr)
        act=self._softmax(ut)
        self._start(act,t)

    def _pressures(self):
        return{
            "hunger": max(0,(self.hunger -self.TH_H)/(100-self.TH_H)),
            "fatigue":max(0,(self.fatigue-self.TH_F)/(100-self.TH_F)),
            "injury": max(0,(self.injury -self.TH_I)/(100-self.TH_I)),
            "fear":   self.fear/100,
        }

    def _utilities(self,pr):
        u=defaultdict(float)
        u["rest"]=pr["fatigue"]*1.3+pr["injury"]*0.5
        if self.species=="human":
            u["eat_grass"]   =pr["hunger"]*(0.6-self.risk_tolerance*0.3)*1.2
            u["eat_berry"]   =pr["hunger"]*(0.7-self.risk_tolerance*0.2)*1.1
            u["eat_fish"]    =pr["hunger"]*0.7
            u["hunt"]        =pr["hunger"]*self.risk_tolerance*1.4
            u["patrol"]      =0.03+self.dopamine*0.04
        elif self.species=="herbivore":
            u["eat_grass"]   =pr["hunger"]*1.5
            u["eat_berry"]   =pr["hunger"]*0.5
            u["patrol"]      =0.03+self.dopamine*0.03
        elif self.species=="predator":
            u["hunt_herb"]   =pr["hunger"]*1.6
            u["rest"]       +=pr["hunger"]*0.2
            u["patrol"]      =0.04+self.dopamine*0.05
        for a in list(u.keys()):
            u[a]*=(1+self.kappa[a]*self.g)
        return u

    def _softmax(self,u):
        temp=max(self.T,0.01)
        names=list(u.keys()); vals=np.array([u[n] for n in names])
        ev=np.exp((vals-vals.max())/temp); probs=ev/ev.sum()
        return np.random.choice(names,p=probs)

    def _start(self,action,t):
        env=self.env
        pos=self.pos()

        if action=="rest":
            r=26*(1+0.2*self.stamina)
            self.fatigue=max(0,self.fatigue-r)
            self.injury =max(0,self.injury -3)
            self.update_kappa("rest",True,r*0.5,"rest")
            mp=max(0,(self.fatigue-self.TH_F)/(100-self.TH_F))
            self.update_heat(mp,r/26)
            self._log(t,"rest"); return

        elif action=="eat_grass":
            # その場に草があれば即食べる
            if pos in env.grass and env.grass[pos]>0.05:
                s,f=env.eat_grass(pos); self._eat(t,"eat_grass",s,f); return
            nodes=env.nearest_passable_resource(pos,env.grass,k=4)
            if nodes: self.target_pos=nodes[0]; self.action_type="eat_grass"; self.state="Moving"

        elif action=="eat_berry":
            if pos in env.berries and env.berries[pos]["abundance"]>0.05:
                s,f=env.eat_berry(pos); self._eat(t,"eat_berry",s,f); return
            nodes=env.nearest_passable_resource(pos,env.berries,k=4)
            if nodes: self.target_pos=nodes[0]; self.action_type="eat_berry"; self.state="Moving"

        elif action=="eat_fish":
            nodes=env.nearest_passable_resource(pos,env.fish,k=4)
            if nodes: self.target_pos=nodes[0]; self.action_type="eat_fish"; self.state="Moving"

        elif action=="hunt":
            preys=self.nearby(radius=10,species_filter=["herbivore"])
            if preys:
                prey=min(preys,key=lambda o:self.dist_to(o.pos()))
                self.target_pos=prey.pos(); self.action_type="hunt"
                self.coop_target=prey; self.state="Moving"

        elif action=="hunt_herb":
            preys=self.nearby(radius=12,species_filter=["herbivore"])
            if preys:
                prey=min(preys,key=lambda o:self.dist_to(o.pos()))
                self.target_pos=prey.pos(); self.action_type="hunt_herb"
                self.coop_target=prey; self.state="Moving"

        elif action=="patrol":
            mr=int(1+self.T*5)
            tx=self.x+random.randint(-mr,mr); ty=self.y+random.randint(-mr,mr)
            self.target_pos=(max(0,min(env.size-1,tx)),max(0,min(env.size-1,ty)))
            self.action_type="patrol"; self.state="Moving"
            self._log(t,"patrol_start"); return

        if self.state!="Moving": self._log(t,"idle")

    def _execute(self,t):
        act=self.action_type; pos=self.pos(); env=self.env

        if act=="eat_grass":
            s,f=env.eat_grass(pos); self._eat(t,"eat_grass",s,f)
        elif act=="eat_berry":
            s,f=env.eat_berry(pos); self._eat(t,"eat_berry",s,f)
        elif act=="eat_fish":
            s,f=env.eat_fish(pos); self._eat(t,"eat_fish",s,f)
        elif act in("hunt","hunt_herb"):
            prey=self.coop_target
            if prey and prey.alive and self.dist_to(prey.pos())<=2:
                base=0.5 if act=="hunt_herb" else 0.45
                p=base*(1-self.injury/100)*(1+self.risk_tolerance*0.3)
                if act=="hunt": p*=self.risk_tolerance
                p=max(0.05,min(0.92,p))
                if random.random()<p:
                    food=random.uniform(30,60)
                    self.hunger=max(0,self.hunger-food)
                    prey.alive=False
                    prey._log(t,"killed",{"killer":self.name})
                    self.update_kappa(act,True,food,act)
                    mp=max(0,(self.hunger-self.TH_H)/(100-self.TH_H))
                    self.update_heat(mp,self.alignment_flow(act,mp))
                    self._log(t,"hunt_success",{"prey":prey.name,"food":round(food,1)})
                else:
                    self.fatigue=min(120,self.fatigue+6)
                    self.injury =min(120,self.injury +2)
                    self.update_kappa(act,False,0,act)
                    self._log(t,"hunt_fail")
            else:
                self._log(t,"hunt_miss")
            self.coop_target=None
        elif act=="patrol":
            self._log(t,"patrol_arrive")
        else:
            self._log(t,"arrived_other")

    def _eat(self,t,action,success,food):
        if success:
            self.hunger=max(0,self.hunger-food)
            self.fatigue=min(120,self.fatigue+0.8)
            mp=max(0,(self.hunger-self.TH_H)/(100-self.TH_H))
            self.update_kappa(action,True,food,action)
            self.update_heat(mp,self.alignment_flow(action,mp))
            self._log(t,f"{action}_ok",{"food":round(food,1)})
        else:
            self.fatigue=min(120,self.fatigue+1.5)
            self.update_kappa(action,False,0,action)
            mp=max(0,(self.hunger-self.TH_H)/(100-self.TH_H))
            self.update_heat(mp,0)
            self._log(t,f"{action}_fail")


class MetaAI:
    def __init__(self,env,roster):
        self.env=env; self.roster=roster; self.log=[]
    def step(self,t):
        env=self.env
        g,b,f=env.resource_summary()
        herbs=[a for a in self.roster.values() if a.alive and a.species=="herbivore"]
        preds=[a for a in self.roster.values() if a.alive and a.species=="predator"]
        inv=[]
        if g<30:
            for p in random.sample(list(env.grass.keys()),min(40,len(env.grass))):
                env.grass[p]=min(1.0,env.grass[p]+0.25)
            inv.append("grass_boost")
        if len(herbs)<4:
            self._spawn("herbivore"); inv.append("herb_spawn")
        if len(preds)>len(herbs)+2:
            for pr in preds: pr.hunger=max(0,pr.hunger-8)
            inv.append("pred_satiate")
        if inv:
            self.log.append({"t":t,"inv":inv,"herbs":len(herbs),"preds":len(preds),"grass":round(g,0)})

    def _spawn(self,species):
        env=self.env
        cells=[(x,y) for y in range(env.size) for x in range(env.size) if env.terrain[y][x]==TERRAIN_PLAIN]
        if cells:
            pos=random.choice(cells)
            preset=random.choice([HERB_S,HERB_M])
            name=f"Herb_sp_{env.t}"
            a=SSDAgent(name,preset,env,self.roster,pos)
            self.roster[name]=a


HUMAN_FORAGER={"species":"human","size_class":"medium",
    "dopamine":0.6,"serotonin":0.65,"oxytocin":0.8,"noradrenaline":0.3,
    "risk_tolerance":0.2,"stamina":0.7,
    "kappa_init":{"eat_grass":0.5,"eat_berry":0.7,"eat_fish":0.4,"rest":0.7}}
HUMAN_HUNTER={"species":"human","size_class":"medium",
    "dopamine":0.8,"serotonin":0.5,"oxytocin":0.5,"noradrenaline":0.5,
    "risk_tolerance":0.75,"stamina":0.8,
    "kappa_init":{"hunt":0.6,"eat_fish":0.5,"rest":0.6}}
HUMAN_GUARDIAN={"species":"human","size_class":"medium",
    "dopamine":0.4,"serotonin":0.7,"oxytocin":0.9,"noradrenaline":0.35,
    "risk_tolerance":0.3,"stamina":0.9,
    "kappa_init":{"eat_grass":0.6,"eat_berry":0.5,"rest":0.8}}
HERB_S={"species":"herbivore","size_class":"small",
    "dopamine":0.35,"serotonin":0.3,"oxytocin":0.7,"noradrenaline":0.92,
    "risk_tolerance":0.08,"stamina":0.5,
    "kappa_init":{"eat_grass":0.8,"rest":0.5}}
HERB_M={"species":"herbivore","size_class":"medium",
    "dopamine":0.3,"serotonin":0.45,"oxytocin":0.6,"noradrenaline":0.75,
    "risk_tolerance":0.15,"stamina":0.7,
    "kappa_init":{"eat_grass":0.7,"eat_berry":0.4,"rest":0.6}}
PRED_S={"species":"predator","size_class":"small",
    "dopamine":0.8,"serotonin":0.55,"oxytocin":0.3,"noradrenaline":0.45,
    "risk_tolerance":0.7,"stamina":0.8,
    "kappa_init":{"hunt_herb":0.8,"rest":0.5}}
PRED_M={"species":"predator","size_class":"medium",
    "dopamine":0.75,"serotonin":0.6,"oxytocin":0.25,"noradrenaline":0.4,
    "risk_tolerance":0.85,"stamina":0.9,
    "kappa_init":{"hunt_herb":0.9,"rest":0.4}}


def run(ticks=600):
    env=Ecosystem(size=40)
    roster={}
    s=env.size

    agents=[
        SSDAgent("Human_Forager", HUMAN_FORAGER, env,roster,(20,25)),
        SSDAgent("Human_Hunter",  HUMAN_HUNTER,  env,roster,(22,23)),
        SSDAgent("Human_Guardian",HUMAN_GUARDIAN,env,roster,(18,27)),
    ]
    for i in range(12):
        while True:
            x,y=random.randint(0,s-1),random.randint(0,s-1)
            if env.terrain[y][x]==TERRAIN_PLAIN: break
        agents.append(SSDAgent(f"Herb_{i}",HERB_S if i<6 else HERB_M,env,roster,(x,y)))
    for i in range(3):
        while True:
            x,y=random.randint(0,s-1),random.randint(0,s-1)
            if env.terrain[y][x]==TERRAIN_PLAIN: break
        agents.append(SSDAgent(f"Pred_{i}",PRED_S if i<2 else PRED_M,env,roster,(x,y)))

    for a in agents: roster[a.name]=a
    meta=MetaAI(env,roster)

    pop={"t":[],"human":[],"herbivore":[],"predator":[]}
    print("=== RDL Ecosystem v2 ===")
    print(f"Agents: {len(agents)} | Ticks: {ticks}")

    for t in range(ticks):
        ags=list(roster.values()); random.shuffle(ags)
        for a in ags: a.step(t)
        env.step(); meta.step(t)
        if t%5==0:
            pop["t"].append(t)
            for sp in["human","herbivore","predator"]:
                pop[sp].append(sum(1 for a in roster.values() if a.alive and a.species==sp))
        if t%100==0:
            c={sp:sum(1 for a in roster.values() if a.alive and a.species==sp) for sp in["human","herbivore","predator"]}
            g,b,f=env.resource_summary()
            print(f"T{t:4d} | {c} | G:{g:.0f} B:{b:.1f} F:{f:.1f}")

    return env,roster,pop,meta


def visualize(env,roster,pop,meta):
    fig=plt.figure(figsize=(20,14))
    gs=fig.add_gridspec(3,3,hspace=0.42,wspace=0.35)

    # 地形マップ
    ax=fig.add_subplot(gs[0:2,0:2])
    img=np.zeros((env.size,env.size,3))
    cm={TERRAIN_PLAIN:[0.78,0.90,0.63],TERRAIN_WATER:[0.39,0.71,0.96],
        TERRAIN_ROCK:[0.56,0.64,0.68],TERRAIN_FOREST:[0.22,0.56,0.24]}
    for y in range(env.size):
        for x in range(env.size):
            img[y,x]=cm[env.terrain[y][x]]
    # 草密度オーバーレイ
    for(x,y),v in env.grass.items():
        img[y,x]=[img[y,x][0]*(1-v*0.25),min(1,img[y,x][1]+v*0.08),img[y,x][2]*(1-v*0.15)]
    ax.imshow(img,origin='upper',interpolation='nearest')
    sp_style={"human":("*",200,"gold"),"herbivore":("o",70,"white"),"predator":("^",100,"red")}
    for a in roster.values():
        if not a.alive: continue
        mk,sz,col=sp_style[a.species]
        ax.scatter(a.x,a.y,marker=mk,s=sz,c=col,edgecolors='black',linewidths=0.8,zorder=3)
    leg=[Patch(facecolor=[0.78,0.90,0.63],label='Plain'),Patch(facecolor=[0.39,0.71,0.96],label='Water'),
         Patch(facecolor=[0.56,0.64,0.68],label='Rock'),Patch(facecolor=[0.22,0.56,0.24],label='Forest'),
         plt.scatter([],[],marker='*',c='gold',edgecolors='k',s=120,label='Human'),
         plt.scatter([],[],marker='o',c='white',edgecolors='k',s=60,label='Herbivore'),
         plt.scatter([],[],marker='^',c='red',edgecolors='k',s=80,label='Predator')]
    ax.legend(handles=leg,loc='upper right',fontsize=7,framealpha=0.85)
    ax.set_title("Ecosystem Map - Final State",fontsize=13,fontweight='bold')
    ax.axis('off')

    # 個体数推移
    ax2=fig.add_subplot(gs[0,2])
    t_arr=pop["t"]
    ax2.plot(t_arr,pop["human"],    label="Human",    color="gold", lw=2)
    ax2.plot(t_arr,pop["herbivore"],label="Herbivore",color="green",lw=2)
    ax2.plot(t_arr,pop["predator"], label="Predator", color="red",  lw=2)
    ax2.set_title("Population",fontsize=11,fontweight='bold')
    ax2.set_xlabel("Tick"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # メタAI介入
    ax3=fig.add_subplot(gs[1,2])
    if meta.log:
        mt=[e["t"] for e in meta.log]
        mg=[e["grass"] for e in meta.log]
        mh=[e["herbs"] for e in meta.log]
        ax3.plot(mt,mg,label="Grass(sum)",color="green",lw=1.5)
        ax3b=ax3.twinx()
        ax3b.plot(mt,mh,label="Herbs alive",color="orange",lw=1.5,linestyle='--')
        ax3b.set_ylabel("Herbivores",color="orange")
    ax3.set_title("Meta-AI Interventions",fontsize=11,fontweight='bold')
    ax3.set_xlabel("Tick"); ax3.set_ylabel("Grass",color="green"); ax3.grid(alpha=0.3)

    # Human Hunger
    ax4=fig.add_subplot(gs[2,0])
    cols=["#e74c3c","#3498db","#2ecc71"]
    humans=[a for a in roster.values() if a.species=="human"]
    for i,a in enumerate(humans):
        if a.log:
            df=pd.DataFrame(a.log)
            ax4.plot(df["t"],df["hunger"],label=a.name.split("_")[1],color=cols[i],lw=1.5)
    ax4.axhline(100,color='red',linestyle='--',alpha=0.5)
    ax4.set_title("Human Hunger",fontsize=11,fontweight='bold')
    ax4.set_xlabel("Tick"); ax4.set_ylabel("Hunger"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # Human E
    ax5=fig.add_subplot(gs[2,1])
    for i,a in enumerate(humans):
        if a.log:
            df=pd.DataFrame(a.log)
            if "E" in df.columns:
                ax5.plot(df["t"],df["E"],label=a.name.split("_")[1],color=cols[i],lw=1.5)
    ax5.set_title("Human Heat (E)",fontsize=11,fontweight='bold')
    ax5.set_xlabel("Tick"); ax5.set_ylabel("E"); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # Human kappa
    ax6=fig.add_subplot(gs[2,2])
    alive_humans=[a for a in roster.values() if a.species=="human" and a.alive]
    if not alive_humans: alive_humans=humans
    for i,a in enumerate(alive_humans):
        top=sorted(a.kappa.items(),key=lambda x:-x[1])[:5]
        labels=[k for k,v in top]; vals=[v for k,v in top]
        xp=np.arange(len(labels))+i*0.28
        ax6.bar(xp,vals,width=0.25,label=a.name.split("_")[1],color=cols[i],alpha=0.8)
    ax6.set_title("Human kappa Top5",fontsize=11,fontweight='bold')
    ax6.set_ylabel("kappa"); ax6.legend(fontsize=8); ax6.grid(alpha=0.3,axis='y')

    plt.suptitle("RDL Ecosystem Simulation v2",fontsize=15,fontweight='bold',y=1.01)
    plt.savefig('/mnt/user-data/outputs/rdl_ecosystem_v2.png',dpi=150,bbox_inches='tight')
    print("Plot saved!")


if __name__=="__main__":
    env,roster,pop,meta=run(ticks=600)
    print("\n=== FINAL ===")
    for sp in["human","herbivore","predator"]:
        ags=[a for a in roster.values() if a.species==sp]
        alive=[a for a in ags if a.alive]
        print(f"{sp:12s}: {len(alive)}/{len(ags)} alive")
    print("\n=== HUMANS ===")
    for a in roster.values():
        if a.species!="human": continue
        st="ALIVE" if a.alive else "DEAD"
        print(f"  {a.name}: {st} | H={a.hunger:.0f} F={a.fatigue:.0f} | E={a.E:.2f} T={a.T:.2f}")
        top=sorted(a.kappa.items(),key=lambda x:-x[1])[:4]
        print(f"    kappa: {[(k,round(v,2)) for k,v in top]}")
    print(f"\nMeta-AI: {len(meta.log)} interventions")
    visualize(env,roster,pop,meta)