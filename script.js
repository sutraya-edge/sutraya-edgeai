const W=23,H=13,WH=W*H,T=64,DIRS=[[0,0],[1,0],[-1,0],[0,1],[0,-1]];

const wall=(x,y)=>y%3!=0&&x>=1&&x<=21&&x!=7&&x!=15,
idx=(x,y)=>y*W+x,
cx=c=>c%W,
cy=c=>c/W|0;

const rng=s=>()=>(s=(s*1664525+1013904223)>>>0)/4294967296;

let ACC=2;

const JT=9,
DOCKS=[idx(4,12),idx(11,12),idx(18,12)],
CHG=[idx(0,12),idx(22,12)],
PICKS=[];

for(const y of [0,3,6,9])
    for(let x=2;x<=20;x++)
        if(x!=7&&x!=15)
            PICKS.push(idx(x,y));

const HOT=PICKS.filter(c=>cy(c)==3||cy(c)==6);

const md=(a,b)=>
    Math.abs(cx(a)-cx(b))+Math.abs(cy(a)-cy(b));

const near=(l,c)=>
    l.reduce((a,b)=>md(b,c)<md(a,c)?b:a);


/* =========================================================
   SIMULATION
   ========================================================= */

function Sim(mode,seed,N,NT,hot){

const R=rng(seed);

const s={
    mode,
    t:0,
    blocks:new Set(),
    tasks:[],
    rs:[],
    done:0,
    estop:0,
    coll:0,
    msgs:0,
    log:[],
    loss:.1,
    range:10,
    NT,
    fin:0,
    cap:40,
    dl:0,
    indl:0,
    lm:0,
    rer:0,
    dist:0,
    jams:0,
    detours:0,
    jl:[],
    j:null,
    jc:0,
    restarts:0,
    slows:0,
    bund:0
};

const L=m=>{
    s.log.unshift('t'+s.t+' '+m);
    s.log.length=Math.min(s.log.length,7);
};

for(let i=0;i<NT;i++)
    s.tasks.push({
        id:i,
        p:(hot?HOT:PICKS)[R()*(hot?HOT:PICKS).length|0],
        st:0,
        d:0,
        w:5+R()*35|0,
        u:1+R()*4|0
    });

const HM=[
    [2,12],
    [8,12],
    [13,12],
    [20,12],
    [10,0],
    [16,0],
    [4,6],
    [18,3]
];

for(let i=0;i<N;i++)
    s.rs.push({
        id:i,
        x:HM[i][0],
        y:HM[i][1],
        task:null,
        path:[],
        bat:55+R()*40|0,
        stall:0,
        wait:0,
        fail:0,
        cool:0,
        known:new Set(),
        bay:0,
        park:0,
        dead:0,
        dock:0,
        dir:0,
        cr:0,
        pz:0,
        delay:0,
        lmv:-1,
        jam:new Map(),
        t2:null,
        slow:0,
        mv:0
    });

const occ=r=>!r.dead&&!r.park&&!r.bay;

const pos=r=>r.y*W+r.x;

s.pri=r=>
    (r.task?2+r.task.u*.6:0)
    +(r.task&&r.task.st==2?4:0)
    +Math.min(r.stall,20)*.4
    +(r.bat<20?6:0)
    -r.id*.001;

const pend=r=>
    [r.task,r.t2].filter(k=>k&&k.st==1);

const dk=r=>
    r.task&&!pend(r).length;

const goal=r=>{
    if(r.task){
        const q=pend(r);

        return q.length
            ? q.reduce(
                (a,b)=>md(b.p,pos(r))<md(a.p,pos(r))?b:a
              ).p
            : r.dock;
    }

    return r.bat<25
        ? near(CHG,pos(r))
        : -1;
};


/* =========================================================
   PATH HELPERS
   ========================================================= */

function spread(p,c){

    const mv=p.filter((x,i)=>x!=(i?p[i-1]:c)),
    m=mv.length,
    L=p.length;

    if(m==L||!m)
        return p;

    const q=[];
    let cur=c,mi=0;

    for(let i=0;i<L;i++){
        const nm=Math.floor((i+1)*m/L);

        if(nm>mi){
            cur=mv[mi];
            mi=nm;
        }

        q.push(cur);
    }

    return q;
}


function bfs(r,O,gf,strict){

    const st=pos(r),
    vis=new Uint8Array((T+1)*WH*3),
    par=new Int16Array((T+1)*WH*3),
    l0=strict&&r.mv?0:2;

    let cur=[st*3+l0];

    vis[st*3+l0]=1;

    for(let t=0;t<T;t++){

        const nx=[];

        for(const e of cur){

            const c=(e/3)|0,
            l=e%3,
            x=cx(c),
            y=cy(c);

            for(const[dx,dy]of DIRS){

                const wt=!dx&&!dy;

                if(wt&&strict&&l==1)
                    continue;

                const X=x+dx,
                Y=y+dy;

                if(
                    X<0||
                    Y<0||
                    X>=W||
                    Y>=H||
                    wall(X,Y)
                )
                    continue;

                const n=Y*W+X;

                if(
                    r.known.has(n)||
                    (!r.ij&&r.jam.get(n)>s.t&&!gf(n))
                )
                    continue;

                const k=(t+1)*WH+n;

                if(O[k])
                    continue;

                const a=O[t*WH+n];

                if(a&&a==O[(t+1)*WH+c])
                    continue;

                const nl=wt
                    ?(l==2?2:1)
                    :0;

                const vk=k*3+nl;

                if(vis[vk])
                    continue;

                vis[vk]=1;

                par[vk]=c*3+l;

                if(gf(n)){

                    const p=[];

                    let m=n,
                    ml=nl;

                    for(let u=t+1;u>0;u--){

                        p.unshift(m);

                        const pv=
                            par[(u*WH+m)*3+ml];

                        m=(pv/3)|0;
                        ml=pv%3;
                    }

                    return p;
                }

                nx.push(n*3+nl);
            }
        }

        cur=nx;
    }

    return null;
}


function resv(ms){

    const O=new Int8Array((T+2)*WH),
    busy=new Uint8Array(WH);

    for(const m of ms){

        let c=m.c0;

        for(let t=0;t<=T+1;t++){

            if(t>0&&t<=m.path.length)
                c=m.path[t-1];

            O[t*WH+c]=m.id+1;
            busy[c]=1;
        }
    }

    return {O,busy};
}


function clash(r,O){

    let c=pos(r);

    const p=r.path,
    l=p.length;

    for(
        let t=1;
        t<=Math.min(T,l+8);
        t++
    ){

        const n=
            t<=l
                ?p[t-1]
                :(p[l-1]??c);

        if(O[t*WH+n])
            return 1;

        const a=O[(t-1)*WH+n];

        if(a&&a==O[t*WH+c])
            return 1;

        c=n;
    }

    return 0;
}


/* =========================================================
   TASK / FAILURE / JAM LOGIC
   ========================================================= */

const rel=(r,why)=>{

    L(
        'R'+(r.id+1)+
        ' '+why+
        ' → tasks re-auctioned'
    );

    for(const k of[r.task,r.t2])
        if(k)
            k.st=0;

    r.task=r.t2=null;
    r.path=[];
    r.cool=8;
    r.fail=0;
};


s.toggle=c=>{

    if(wall(cx(c),cy(c)))
        return;

    if(s.blocks.has(c)){

        s.blocks.delete(c);

        s.rs.forEach(r=>{
            r.known.delete(c);
            r.path=[];
        });

        L('cell cleared');

    }else if(
        !s.rs.some(
            r=>occ(r)&&pos(r)==c
        )
    ){

        s.blocks.add(c);
        L('aisle cell blocked');
    }
};


s.declare=(r,c)=>{

    if(r.jam.get(c)>s.t)
        return;

    const x=cx(c),
    y=cy(c);

    for(const[dx,dy]of DIRS)

        if(
            x+dx>=0&&
            y+dy>=0&&
            x+dx<W&&
            y+dy<H&&
            !wall(x+dx,y+dy)
        )

            r.jam.set(
                idx(x+dx,y+dy),
                s.t+JT
            );

    s.jams++;
    s.jl.push(s.t+JT);

    L(
        'R'+(r.id+1)+
        ' senses traffic jam at ('+
        x+','+y+
        '), tells all: avoid '+
        JT+' s'
    );

    r.path=[];
};


s.mkjam=()=>{

    if(mode!='hive')
        return;

    const a=s.rs.filter(
        r=>occ(r)&&r.path.length
    );

    if(a.length){

        const r=
            a[R()*a.length|0];

        s.declare(
            r,
            r.path[
                Math.min(
                    2,
                    r.path.length-1
                )
            ]
        );
    }
};


s.setup=k=>{

    s.cap=0;

    s.tasks.forEach((t,i)=>{

        const u=
            (k==3?[5,3,1]:[5,2,3])[i];

        if(u)
            t.u=u;
    });

    const P=(i,x,y)=>{

        const r=s.rs[i];

        r.x=x;
        r.y=y;
        r.bat=90;
    };

    const Q=(i,x,y)=>{

        const t=s.tasks[i];

        t.p=idx(x,y);
        t.st=1;

        s.rs[i].task=t;

        P(
            i,
            s.rs[i].x,
            s.rs[i].y
        );
    };


    if(k==3){

        P(0,7,7);
        P(1,7,11);
        P(2,5,9);

        [9,10,11].forEach(
            (x,i)=>
                s.tasks[i].p=idx(x,9)
        );

        s.j=idx(7,9);

        L(
            '3-way test: R1 (north), R2 (south), R3 (west) all reach the junction together'
        );

    }else{

        P(0,3,0);
        P(1,11,0);
        P(2,7,0);

        Q(0,12,0);
        Q(1,2,0);
        Q(2,20,0);

        L(
            'Head-on test: R1, R2 and R3 meet in a single lane'
        );
    }
};


s.kill=()=>{

    const a=s.rs.filter(occ);

    if(a.length<2)
        return;

    const r=
        a[R()*a.length|0];

    r.dead=s.t;

    s.blocks.add(pos(r));

    L(
        'R'+r.id+
        ' FAILED in aisle'
    );
};


/* =========================================================
   SIMULATION STEP
   ========================================================= */

s.step=function(){

    if(s.fin)
        return;

    s.t++;

    const rs=s.rs,
    live=rs.filter(occ);


    for(const r of rs)
        if(r.cool>0)
            r.cool--;


    for(const r of rs)
        for(const[k,e]of r.jam)
            if(e<=s.t)
                r.jam.delete(k);


    while(
        s.jl.length&&
        s.jl[0]<=s.t
    ){

        s.jl.shift();

        L(
            'jam cleared: normal routes reopen'
        );
    }


    for(const r of live){

        for(const b of s.blocks)

            if(
                md(pos(r),b)<=2&&
                !r.known.has(b)
            ){

                r.known.add(b);
                r.path=[];

                L(
                    'R'+r.id+
                    ' senses block'
                );
            }
    }


    s.msgs+=live.length;


    for(const a of live)
        for(const b of live)

            if(
                a!=b&&
                md(pos(a),pos(b))<=s.range
            ){

                for(const k of a.known)

                    if(
                        !b.known.has(k)&&
                        R()>s.loss
                    ){

                        b.known.add(k);
                        b.path=[];
                    }


                for(const[k,e]of a.jam)

                    if(
                        e>s.t&&
                        !(b.jam.get(k)>s.t)&&
                        R()>s.loss
                    ){

                        b.jam.set(k,e);

                        if(b.path.includes(k)){
                            b.path=[];
                            s.detours++;
                        }
                    }
            }


    for(const r of rs){

        if(
            r.dead&&
            r.task&&
            s.t-r.dead>=6
        )
            rel(
                r,
                'heartbeat lost'
            );


        if(r.bay){

            r.bat+=3;

            if(
                r.bat>=90&&
                !rs.some(
                    o=>
                        o!=r&&
                        occ(o)&&
                        pos(o)==pos(r)
                )
            )
                r.bay=0;
        }
    }


    for(const r of rs.filter(occ)){

        const g=goal(r);

        if(
            g>=0&&
            (
                dk(r)
                    ?md(g,pos(r))<=1
                    :g==pos(r)
            )
        ){

            if(r.task){

                if(dk(r)){

                    for(const k of[r.task,r.t2])
                        if(k){

                            k.st=3;
                            s.done++;
                        }

                    r.task=r.t2=null;

                }else{

                    for(const k of[r.task,r.t2])
                        if(
                            k&&
                            k.st==1&&
                            k.p==pos(r)
                        )
                            k.st=2;

                    r.dock=
                        near(
                            DOCKS,
                            pos(r)
                        );
                }

                r.path=[];
                r.mv=0;

            }else{

                r.bay=1;
            }
        }
    }


    const idleN=
        rs.filter(
            r=>
                occ(r)&&
                !r.task&&
                r.bat>=25&&
                !r.cool
        ).length;


    for(
        const t of s.tasks.filter(
            t=>t.st==0
        )
    ){

        let host=null,
        hd=idleN?2:3;

        if(s.cap>0)

            for(const r of rs)

                if(
                    occ(r)&&
                    r.task&&
                    !r.t2&&
                    r.task.w+t.w<=s.cap&&
                    r.bat>=25
                ){

                    const d=
                        r.path
                            .concat(goal(r))
                            .reduce(
                                (m,c)=>
                                    c>=0
                                        ?Math.min(
                                            m,
                                            md(c,t.p)
                                        )
                                        :m,
                                9
                            );

                    if(d<hd){

                        hd=d;
                        host=r;
                    }
                }


        if(host){

            host.t2=t;
            t.st=1;
            host.path=[];
            s.bund++;

            L(
                'R'+(host.id+1)+
                ' bundles task '+t.id+
                ' en route ('+
                (host.task.w+t.w)+
                ' of '+s.cap+
                ' kg payload)'
            );
        }
    }


    for(;;){

        const P=
            s.tasks.filter(
                t=>t.st==0
            );

        const I=
            rs.filter(
                r=>
                    occ(r)&&
                    !r.task&&
                    r.bat>=25&&
                    !r.cool
            );

        if(!P.length||!I.length)
            break;

        let bt=null,
        br=null,
        bb=1e9;

        for(const t of P)
            for(const r of I){

                const b=
                    md(pos(r),t.p)+
                    (100-r.bat)*.1;

                if(b<bb){

                    bb=b;
                    bt=t;
                    br=r;
                }
            }


        br.task=bt;
        bt.st=1;
        br.path=[];
        br.fail=0;
    }


    for(const r of rs)

        if(
            occ(r)&&
            !r.task&&
            r.bat>=25&&
            !r.cool
        ){

            let bt=null,
            bh=null,
            bg=4;

            for(const h of rs)

                if(
                    h!=r&&
                    occ(h)&&
                    h.task&&
                    h.task.st==1&&
                    (!h.t2||h.t2.st==1)
                )

                    for(const t of[h.task,h.t2])

                        if(t){

                            const g=
                                md(pos(h),t.p)-
                                md(pos(r),t.p);

                            if(g>bg){

                                bg=g;
                                bt=t;
                                bh=h;
                            }
                        }


            if(bt){

                if(bh.task==bt){

                    bh.task=bh.t2;
                    bh.t2=null;

                }else{

                    bh.t2=null;
                }

                bh.path=[];

                r.task=bt;
                r.path=[];
                r.fail=0;

                s.swaps=(s.swaps||0)+1;

                if(s.swaps%4==1)

                    L(
                        'R'+(r.id+1)+
                        ' takes task '+bt.id+
                        ' from R'+
                        (bh.id+1)+
                        ' (closer robot)'
                    );
            }
        }


    const pool=
        s.tasks.some(
            t=>t.st==0
        );


    for(const r of rs)

        if(
            !r.dead&&
            !r.bay&&
            !r.task&&
            r.bat>=25
        ){

            const np=pool?0:1;

            if(
                np||
                !r.park||
                !rs.some(
                    o=>
                        o!=r&&
                        occ(o)&&
                        pos(o)==pos(r)
                )
            )
                r.park=np;
        }


    const act=rs.filter(occ);


    /* =====================================================
       HIVE LANE P2P COORDINATION
       ===================================================== */

    if(mode=='hive'){

        const ms=
            act.map(
                r=>({
                    id:r.id,
                    c0:pos(r),
                    path:r.path.slice(),
                    pri:s.pri(r)
                })
            );


        for(const r of act){

            if(r.yl>0){

                r.yl--;
                continue;
            }


            const pr=s.pri(r);

            const heard=
                ms.filter(
                    m=>
                        m.id!=r.id&&
                        m.pri>pr&&
                        md(m.c0,pos(r))<=s.range&&
                        R()>s.loss
                );


            const {O,busy}=resv(heard);

            const g=goal(r);

            let need=0;


            if(!r.path.length){

                if(
                    g>=0&&
                    !(
                        dk(r)
                            ?md(g,pos(r))<=1
                            :g==pos(r)
                    )
                )
                    need=1;
                else if(clash(r,O))
                    need=2;

            }else if(clash(r,O)){

                need=
                    g>=0
                        ?1
                        :2;
            }


            if(need){

                const gf=
                    need==1
                        ?(
                            dk(r)
                                ?n=>md(n,g)<=1
                                :n=>n==g
                        )
                        :n=>!busy[n];


                let p=
                    bfs(
                        r,
                        O,
                        gf,
                        1
                    )||
                    bfs(
                        r,
                        O,
                        gf
                    );


                if(!p){

                    r.ij=1;

                    p=bfs(
                        r,
                        O,
                        gf
                    );

                    r.ij=0;
                }


                if(p){

                    if(r.path.length)
                        s.rer++;

                    r.path=p;
                    r.slow=0;
                    r.fail=0;

                    const q=
                        spread(
                            p,
                            pos(r)
                        );


                    if(q!==p){

                        r.path=q;

                        if(clash(r,O))
                            r.path=p;

                        else{

                            r.slow=1;

                            if(s.slows++%6==0)

                                L(
                                    'R'+(r.id+1)+
                                    ' slows down, lets higher-priority pass'
                                );
                        }
                    }

                }else if(
                    ++r.fail>12&&
                    r.task&&
                    need==1
                ){

                    rel(
                        r,
                        'no route'
                    );
                }
            }
        }

    }else{

        /* =================================================
           STOP AND WAIT
           ================================================= */

        for(const r of act){

            const g=goal(r);

            if(
                g<0||
                g==pos(r)||
                (
                    dk(r)&&
                    md(g,pos(r))<=1
                )
            )
                continue;


            r.th=r.th||4;

            const w=r.wait>=r.th;

            if(!r.path.length||w){

                let O=
                    new Int8Array(
                        (T+2)*WH
                    );


                if(w)

                    O=
                        resv(
                            act
                                .filter(o=>o!=r)
                                .map(
                                    o=>({
                                        id:o.id,
                                        c0:pos(o),
                                        path:[]
                                    })
                                )
                        ).O;


                let p=
                    bfs(
                        r,
                        O,
                        dk(r)
                            ?n=>md(n,g)<=1
                            :n=>n==g
                    );


                if(!p&&w){

                    const nb=
                        DIRS
                            .slice(1)
                            .filter(
                                d=>
                                    r.x+d[0]>=0&&
                                    r.x+d[0]<W&&
                                    r.y+d[1]>=0&&
                                    r.y+d[1]<H
                            )
                            .map(
                                d=>
                                    idx(
                                        r.x+d[0],
                                        r.y+d[1]
                                    )
                            )
                            .filter(
                                n=>
                                    !wall(
                                        n%W,
                                        n/W|0
                                    )&&
                                    !s.blocks.has(n)&&
                                    !act.some(
                                        o=>pos(o)==n
                                    )
                            );


                    if(nb.length)
                        p=[
                            nb[
                                R()*nb.length|0
                            ]
                        ];
                }


                if(p){

                    r.path=p;

                    if(w){

                        s.rer++;

                        r.wait=0;

                        r.th=
                            5+R()*6|0;
                    }
                }
            }
        }
    }


    /* =====================================================
       MOVEMENT
       ===================================================== */

    const order=
        act.slice().sort(
            mode=='hive'
                ?(a,b)=>s.pri(b)-s.pri(a)
                :(a,b)=>a.id-b.id
        );


    const stop=r=>{

        if(!r.slow&&r.mv){

            r.pz++;

            if(r.pz==2){

                s.restarts++;

                r.bat-=.25;
                r.mv=0;
                r.rs=1;
            }
        }
    };


    const go=(r,n)=>{

        if(r.rs){

            r.rs=0;
            r.ac=ACC;
        }

        if(r.ac>0){

            r.ac--;

            return;
        }

        if(
            r.mv&&
            r.pz==1&&
            !r.slow
        )
            s.slows++;


        s.lm=s.t;
        s.dist++;

        r.lmv=s.t;
        r.pz=0;

        r.dir=
            Math.atan2(
                cy(n)-r.y,
                cx(n)-r.x
            );

        r.x=cx(n);
        r.y=cy(n);

        r.path.shift();

        r.bat-=.06;
        r.stall=0;
        r.mv=1;
    };


    const blk=[];


    for(const r of order){

        const g=goal(r),
        p=r.path;


        if(!p.length){

            r.slow=0;

            if(
                g>=0&&
                g!=pos(r)
            ){

                r.stall++;
                stop(r);
            }

            continue;
        }


        const n=p[0];


        if(n==pos(r)){

            p.shift();

            if(!r.slow){

                r.stall++;
                stop(r);
            }

            continue;
        }


        if(s.blocks.has(n)){

            r.known.add(n);
            r.path=[];
            s.estop++;

            stop(r);

            continue;
        }


        if(
            order.some(
                o=>
                    o!=r&&
                    pos(o)==n
            )
        )
            blk.push(r);

        else
            go(r,n);
    }


    for(
        let k=0;
        k<5&&blk.length;
        k++
    ){

        let mv=0;

        for(
            let i=blk.length-1;
            i>=0;
            i--
        ){

            const r=blk[i],
            n=r.path[0];

            if(
                !order.some(
                    o=>
                        o!=r&&
                        pos(o)==n
                )
            ){

                go(r,n);

                blk.splice(i,1);

                mv=1;
            }
        }

        if(!mv)
            break;
    }


    for(const r of blk){

        const n=r.path[0];

        s.estop++;
        r.wait++;
        r.stall++;

        stop(r);

        if(mode=='hive'){

            if(
                r.stall>=3&&
                order.filter(
                    o=>
                        o!=r&&
                        md(pos(o),n)<=3
                ).length>=2
            )
                s.declare(r,n);

            else
                r.path.unshift(pos(r));
        }
    }


    for(const r of act){

        const g=goal(r);

        if(
            g>=0&&
            g!=pos(r)&&
            r.lmv!=s.t
        )
            r.delay++;
    }


    const wk=
        act.filter(
            r=>{
                const g=goal(r);

                return g>=0&&
                       g!=pos(r);
            }
        );


    if(
        wk.length>=2&&
        s.t-s.lm>=6
    ){

        if(!s.indl){

            s.dl++;
            s.indl=1;

            L(
                'DEADLOCK: nobody moved for 6 s'
            );
        }

    }else{

        s.indl=0;
    }


    if(
        mode=='hive'&&
        wk.length>=2&&
        s.t-s.lm>=8
    ){

        const cand=[];

        for(const r of wk){

            const fr=
                DIRS
                    .slice(1)
                    .map(
                        d=>[
                            r.x+d[0],
                            r.y+d[1]
                        ]
                    )
                    .filter(
                        ([x,y])=>
                            x>=0&&
                            y>=0&&
                            x<W&&
                            y<H&&
                            !wall(x,y)&&
                            !s.blocks.has(
                                idx(x,y)
                            )&&
                            !act.some(
                                o=>
                                    o!=r&&
                                    o.x==x&&
                                    o.y==y
                            )
                    );

            if(fr.length)
                cand.push([r,fr]);
        }


        if(cand.length){

            const[
                r,
                fr
            ]=
                cand[
                    R()*cand.length|0
                ];

            const[
                x,
                y
            ]=
                fr[
                    R()*fr.length|0
                ];

            r.path=[
                idx(x,y)
            ];

            r.yl=8;

            s.lm=s.t-2;

            L(
                'R'+(r.id+1)+
                ' backs out to let others pass'
            );
        }
    }


    const a2=rs.filter(occ);

    for(
        let i=0;
        i<a2.length;
        i++
    )
        for(
            let j=i+1;
            j<a2.length;
            j++
        )
            if(
                pos(a2[i])==
                pos(a2[j])
            )
                s.coll++;


    if(s.j!=null)

        for(const r of rs)

            if(
                occ(r)&&
                pos(r)==s.j&&
                !r.cr
            ){

                r.cr=++s.jc;

                L(
                    'R'+(r.id+1)+
                    ' crosses the junction #'+
                    r.cr+
                    (r.slow
                        ?' (slowed down)'
                        :'')
                );
            }


    if(s.done>=NT)
        s.fin=s.t;
};


return s;
}


/* =========================================================
   BENCHMARK
   ========================================================= */

function bench(seed,N,cap=40,hot=0){

    const o={};

    for(
        const[k,m,c]
        of[
            ['hive','hive',cap],
            ['stop','stop',cap],
            ['trad','stop',0]
        ]
    ){

        const s=
            Sim(
                m,
                seed,
                N,
                N*7,
                hot
            );

        s.cap=c;

        while(
            !s.fin&&
            s.t<2500
        )
            s.step();

        o[k]=s;
    }

    return o;
}


/* =========================================================
   EDGE AI
   =========================================================

   The model was trained in Python and exported to:

       edge_ai/model.json

   The browser loads that model and performs inference locally.
   No server or cloud AI API is used.
   ========================================================= */

let EDGE_MODEL=null;
let EDGE_MODEL_LOADING=null;


async function loadEdgeAIModel(){

    if(EDGE_MODEL)
        return EDGE_MODEL;

    if(EDGE_MODEL_LOADING)
        return EDGE_MODEL_LOADING;


    EDGE_MODEL_LOADING=
        fetch("edge_ai/model.json")
            .then(response=>{

                if(!response.ok)
                    throw new Error(
                        "Could not load edge_ai/model.json"
                    );

                return response.json();
            })
            .then(model=>{

                EDGE_MODEL=model;

                console.log(
                    "HiveLane Edge-AI model loaded:",
                    model.model
                );

                return model;
            })
            .catch(error=>{

                console.error(
                    "Edge-AI model loading failed:",
                    error
                );

                EDGE_MODEL=null;

                return null;
            });


    return EDGE_MODEL_LOADING;
}


/* =========================================================
   EDGE AI FEATURES
   ========================================================= */

function edgeFeatures(s){

    const robots=
        s.rs.filter(
            r=>
                !r.dead&&
                !r.park&&
                !r.bay
        );


    const activeRobots=robots.length;

    let nearbyRobots=0;
    let minimumDistance=99;
    let predictedConflicts=0;
    let junctionPressure=0;
    let headOnPairs=0;
    let waitSum=0;
    let maxUrgency=1;
    let lowBatteryCount=0;
    let jammedCells=0;


    for(
        let i=0;
        i<robots.length;
        i++
    ){

        const a=robots[i];

        waitSum+=a.wait||0;


        if(a.bat<25)
            lowBatteryCount++;


        if(a.task)

            maxUrgency=
                Math.max(
                    maxUrgency,
                    a.task.u||1
                );


        for(
            let j=i+1;
            j<robots.length;
            j++
        ){

            const b=robots[j];

            const d=
                md(
                    pos(a),
                    pos(b)
                );


            minimumDistance=
                Math.min(
                    minimumDistance,
                    d
                );


            if(d<=6)
                nearbyRobots++;


            /* Predict possible conflict
               during next 10 seconds */

            const horizon=10;


            for(
                let t=0;
                t<=horizon;
                t++
            ){

                const pa=
                    t===0
                        ?pos(a)
                        :(
                            a.path[t-1]??
                            a.path[
                                a.path.length-1
                            ]??
                            pos(a)
                        );


                const pb=
                    t===0
                        ?pos(b)
                        :(
                            b.path[t-1]??
                            b.path[
                                b.path.length-1
                            ]??
                            pos(b)
                        );


                if(
                    md(pa,pb)<=1
                ){

                    predictedConflicts++;

                    break;
                }
            }


            /* Head-on movement */

            if(
                a.path.length>0&&
                b.path.length>0
            ){

                const an=a.path[0];
                const bn=b.path[0];

                if(
                    an===pos(b)&&
                    bn===pos(a)
                )
                    headOnPairs++;
            }
        }
    }


    /* Junction pressure */

    const junctionCells=[
        idx(7,9),
        idx(15,9),
        idx(7,6),
        idx(15,6)
    ];


    for(const r of robots){

        if(
            junctionCells.includes(
                pos(r)
            )
        )
            junctionPressure+=0.25;


        for(
            const c of r.path.slice(0,5)
        ){

            if(
                junctionCells.includes(c)
            )
                junctionPressure+=0.15;
        }
    }


    junctionPressure=
        Math.min(
            1,
            junctionPressure
        );


    /* Jammed cells */

    const jamSet=new Set();


    for(const r of s.rs){

        if(r.dead)
            continue;


        for(
            const[cell,expiry]
            of r.jam
        ){

            if(expiry>s.t)
                jamSet.add(cell);
        }
    }


    jammedCells=jamSet.size;


    const averageWait=
        activeRobots>0
            ?waitSum/activeRobots
            :0;


    const lowBatteryRatio=
        activeRobots>0
            ?lowBatteryCount/activeRobots
            :0;


    return[
        activeRobots,
        nearbyRobots,
        minimumDistance===99
            ?8
            :minimumDistance,
        predictedConflicts,
        junctionPressure,
        headOnPairs,
        averageWait,
        maxUrgency,
        lowBatteryRatio,
        jammedCells
    ];
}


/* =========================================================
   RUN THE DECISION TREE LOCALLY
   ========================================================= */

function runEdgeDecisionTree(
    model,
    features
){

    let nodeIndex=0;


    while(true){

        const node=
            model.nodes[nodeIndex];


        if(!node){

            return{
                label:"LOW",
                probability:0,
                probabilities:[
                    1,0,0
                ]
            };
        }


        /* Leaf node */

        if(node.leaf){

            const p=
                node.probabilities||
                [1,0,0];


            let best=0;


            for(
                let i=1;
                i<p.length;
                i++
            ){

                if(
                    p[i]>p[best]
                )
                    best=i;
            }


            return{
                label:model.classes[best],
                probability:p[best],
                probabilities:p
            };
        }


        /* Decision node */

        const value=
            features[node.feature];


        nodeIndex=
            value<=node.threshold
                ?node.left
                :node.right;
    }
}


/* =========================================================
   EDGE AI PREDICTION
   ========================================================= */

function edgeAIPredict(s){

    const start=
        performance.now();


    if(!EDGE_MODEL){

        return{
            lvl:"LOADING",
            probability:0,
            best:null,
            ms:
                performance.now()-start,
            features:[]
        };
    }


    const features=
        edgeFeatures(s);


    const prediction=
        runEdgeDecisionTree(
            EDGE_MODEL,
            features
        );


    /* Find the earliest predicted
       close approach for display */

    const robots=
        s.rs.filter(
            r=>
                !r.dead&&
                !r.park&&
                !r.bay
        );


    let best=null;


    for(
        let i=0;
        i<robots.length;
        i++
    ){

        for(
            let j=i+1;
            j<robots.length;
            j++
        ){

            const a=robots[i];
            const b=robots[j];


            for(
                let t=0;
                t<=10;
                t++
            ){

                const pa=
                    t===0
                        ?pos(a)
                        :(
                            a.path[t-1]??
                            a.path[
                                a.path.length-1
                            ]??
                            pos(a)
                        );


                const pb=
                    t===0
                        ?pos(b)
                        :(
                            b.path[t-1]??
                            b.path[
                                b.path.length-1
                            ]??
                            pos(b)
                        );


                if(
                    md(pa,pb)<=1
                ){

                    if(
                        !best||
                        t<best.t
                    ){

                        best={
                            a:a.id,
                            b:b.id,
                            t:t
                        };
                    }

                    break;
                }
            }
        }
    }


    return{
        lvl:prediction.label,
        probability:
            prediction.probability,
        best:best,
        ms:
            performance.now()-start,
        features:features,
        probabilities:
            prediction.probabilities
    };
}


/* Keep the original function name
   so the rest of the website continues
   working exactly as before. */

function risk(s){

    return edgeAIPredict(s);
}


/* Start loading the model */

loadEdgeAIModel();


/* =========================================================
   USER INTERFACE
   ========================================================= */

if(typeof document!=='undefined'){

const $=q=>document.querySelector(q);

const cv=$('#cv'),
g=cv.getContext('2d'),
CS=30;

const COL=[
    '#22d3ee',
    '#f472b6',
    '#facc15',
    '#a78bfa',
    '#fb923c',
    '#4ade80',
    '#60a5fa',
    '#f87171'
];


cv.width=W*CS;
cv.height=H*CS;


let mode='hive',
sim,
run=true,
seed=7,
iv,
res={},
cmp=0,
sig='',
view='home';


const sg=()=>
    seed+
    '|'+view+
    '|'+ACC+
    '|'+$('#nr').value+
    '|'+$('#cp').value;


function note(){

    if(
        cmp==1&&
        !sim.fin&&
        sim.t>=900
    ){

        if(sig!=sg()){

            res={};
            sig=sg();
        }

        res.stop=-1;
        cmp=2;
        mode='hive';

        $('#md').value='hive';

        reset();

        return;
    }


    if(
        sim.fin&&
        !sim.rec
    ){

        sim.rec=1;

        if(sig!=sg()){

            res={};
            sig=sg();
        }


        res[sim.mode]=sim.fin;


        if(cmp==1){

            cmp=2;

            mode='hive';

            $('#md').value='hive';

            reset();

        }else if(cmp==2){

            cmp=0;
        }
    }
}


function cmpUI(){

    const r=
        sig==sg()
            ?res
            :{};


    const f=v=>
        v
            ?(
                v<0
                    ?'never finished (stuck)'
                    :v+' s'
            )
            :'not run';


    const sv=
        r.stop<0&&r.hive
            ?'stop-and-wait never finished'
            :r.stop&&r.hive
                ?(
                    (
                        1-r.hive/r.stop
                    )*100
                ).toFixed(0)+
                '% faster ('+
                (
                    r.stop-r.hive
                )+
                ' s saved)'
                :'-';


    $('#cmpo').innerHTML=`
<table>
<tr>
<td>1. Stop-and-wait</td>
<td>
${
    cmp==1
        ?'running… '+sim.t+' s'
        :f(r.stop)
}
</td>
</tr>

<tr>
<td>2. HiveLane P2P</td>
<td>
${
    cmp==2
        ?'running… '+sim.t+' s'
        :f(r.hive)
}
</td>
</tr>

<tr>
<td><b>Time saved</b></td>
<td><b>${sv}</b></td>
</tr>
</table>`;
}


/* =========================================================
   RESET
   ========================================================= */

function reset(){

    const N=+$('#nr').value;


    sim=
        view=='dl'
            ?Sim(
                mode,
                seed,
                3,
                3
            )
            :view=='co'
                ?Sim(
                    mode,
                    seed,
                    3,
                    3
                )
                :Sim(
                    mode,
                    seed,
                    N,
                    N*7
                );


    sim.loss=
        +$('#ls').value/100;

    sim.cap=
        +$('#cp').value;


    if(view=='dl')
        sim.setup(3);

    else if(view=='co')
        sim.setup(2);


    draw();
}


/* =========================================================
   DRAW
   ========================================================= */

function draw(){

    const s=sim;

    g.clearRect(
        0,
        0,
        cv.width,
        cv.height
    );


    /* Grid */

    for(
        let y=0;
        y<H;
        y++
    )

        for(
            let x=0;
            x<W;
            x++
        ){

            const c=idx(x,y);


            if(wall(x,y)){

                g.fillStyle='#1e293b';

                g.fillRect(
                    x*CS+2,
                    y*CS+2,
                    CS-4,
                    CS-4
                );
            }


            if(s.blocks.has(c)){

                g.fillStyle='#ef4444';

                g.fillRect(
                    x*CS+3,
                    y*CS+3,
                    CS-6,
                    CS-6
                );
            }


            if(DOCKS.includes(c)){

                g.fillStyle='#22c55e';

                g.fillRect(
                    x*CS+4,
                    y*CS+4,
                    CS-8,
                    CS-8
                );
            }


            if(CHG.includes(c)){

                g.fillStyle='#eab308';

                g.fillRect(
                    x*CS+4,
                    y*CS+4,
                    CS-8,
                    CS-8
                );
            }
        }


    /* Junction */

    if(s.j!=null){

        g.strokeStyle='#38bdf8';
        g.lineWidth=2;

        g.strokeRect(
            cx(s.j)*CS+2,
            cy(s.j)*CS+2,
            CS-4,
            CS-4
        );
    }


    /* Traffic jams */

    const jm=new Map();


    for(const r of s.rs)

        for(
            const[k,e]
            of r.jam
        )

            if(e>s.t)
                jm.set(
                    k,
                    Math.max(
                        jm.get(k)||0,
                        e
                    )
                );


    for(
        const[k,e]
        of jm
    ){

        g.fillStyle=
            'rgba(251,146,60,.4)';

        g.fillRect(
            cx(k)*CS+1,
            cy(k)*CS+1,
            CS-2,
            CS-2
        );


        g.fillStyle='#fff';
        g.font='9px sans-serif';
        g.textAlign='center';

        g.fillText(
            e-s.t,
            cx(k)*CS+CS/2,
            cy(k)*CS+CS/2+3
        );
    }


    $('#m12').textContent=
        [...jm.values()]
            .reduce(
                (a,e)=>
                    Math.max(
                        a,
                        e-s.t
                    ),
                0
            );


    /* Tasks */

    for(const t of s.tasks)

        if(t.st<2){

            g.fillStyle=
                t.st
                    ?'#fb923c'
                    :'#f59e0b';

            g.beginPath();

            g.arc(
                cx(t.p)*CS+CS/2,
                cy(t.p)*CS+CS/2,
                4,
                0,
                7
            );

            g.fill();

            g.fillStyle='#fde68a';
            g.font='9px sans-serif';
            g.textAlign='center';

            g.fillText(
                t.w,
                cx(t.p)*CS+CS/2,
                cy(t.p)*CS+CS/2-6
            );
        }


    /* P2P links */

    const lv=
        s.rs.filter(
            r=>!r.dead&&!r.park
        );


    g.lineWidth=1;
    g.strokeStyle=
        'rgba(45,212,191,.25)';


    for(const a of lv)

        for(const b of lv)

            if(
                a.id<b.id&&
                md(
                    a.y*W+a.x,
                    b.y*W+b.x
                )<=s.range
            ){

                g.beginPath();

                g.moveTo(
                    a.x*CS+CS/2,
                    a.y*CS+CS/2
                );

                g.lineTo(
                    b.x*CS+CS/2,
                    b.y*CS+CS/2
                );

                g.stroke();
            }


    /* Robots */

    for(const r of s.rs){

        const col=COL[r.id];


        if(
            r.path.length&&
            !r.dead
        ){

            g.strokeStyle=col;
            g.globalAlpha=.6;
            g.lineWidth=3;

            g.beginPath();

            g.moveTo(
                r.x*CS+CS/2,
                r.y*CS+CS/2
            );


            for(
                const c of r.path
            )

                g.lineTo(
                    cx(c)*CS+CS/2,
                    cy(c)*CS+CS/2
                );


            g.stroke();

            g.globalAlpha=1;
        }


        g.globalAlpha=
            r.park||r.bay
                ? .45
                :1;


        const X=
            r.x*CS+CS/2;

        const Y=
            r.y*CS+CS/2;


        g.save();

        g.translate(X,Y);

        g.rotate(r.dir);

        g.fillStyle='#020617';


        for(
            const a of[-1,1]
        )

            for(
                const b of[-1,1]
            )

                g.fillRect(
                    a*7-3,
                    b*9-1.5,
                    6,
                    3
                );


        g.fillStyle=
            r.dead
                ?'#64748b'
                :col;

        g.strokeStyle=
            'rgba(0,0,0,.5)';

        g.lineWidth=1;

        g.beginPath();

        if(g.roundRect)

            g.roundRect(
                -11,
                -8,
                22,
                16,
                4
            );

        else
            g.rect(
                -11,
                -8,
                22,
                16
            );

        g.fill();
        g.stroke();


        g.fillStyle=
            'rgba(255,255,255,.28)';

        g.fillRect(
            -8,
            -5,
            12,
            10
        );


        g.fillStyle='#e2e8f0';

        g.fillRect(
            10,
            -6,
            2.5,
            12
        );


        g.fillStyle='#0f172a';

        g.beginPath();

        g.arc(
            6,
            0,
            3,
            0,
            7
        );

        g.fill();


        g.fillStyle=
            r.dead
                ?'#ef4444'
                :'#67e8f9';

        g.beginPath();

        g.arc(
            6,
            0,
            1.2,
            0,
            7
        );

        g.fill();


        g.fillStyle=
            r.dead
                ?'#ef4444'
                :r.slow
                    ?'#f59e0b'
                    :'#22c55e';


        g.beginPath();

        g.arc(
            -9,
            0,
            2,
            0,
            7
        );

        g.fill();


        [
            r.task,
            r.t2
        ]
        .filter(
            k=>k&&k.st==2
        )
        .forEach(
            (k,i)=>{

                g.fillStyle='#b45309';

                g.fillRect(
                    -8,
                    -7+i*8,
                    8,
                    7
                );

                g.fillStyle='#fde68a';

                g.fillRect(
                    -4.5,
                    -7+i*8,
                    1,
                    7
                );
            }
        );


        g.restore();


        g.fillStyle='#fff';
        g.font=
            'bold 9px sans-serif';
        g.textAlign='center';


        g.fillText(
            r.dead
                ?'✕'
                :r.id+1,
            X,
            Y-11
        );


        /* Battery bar */

        g.fillStyle='#1e293b';

        g.fillRect(
            X-8,
            Y+11,
            16,
            2.5
        );


        g.fillStyle=
            r.bat<25
                ?'#ef4444'
                :'#22c55e';


        g.fillRect(
            X-8,
            Y+11,
            16*
            Math.max(
                0,
                Math.min(
                    100,
                    r.bat
                )
            )/100,
            2.5
        );


        if(r.slow){

            g.strokeStyle='#fff';
            g.setLineDash([3,3]);

            g.beginPath();

            g.arc(
                X,
                Y,
                15,
                0,
                7
            );

            g.stroke();

            g.setLineDash([]);
        }


        g.globalAlpha=1;
    }


    /* Metrics */

    $('#m1').textContent=s.t;
    $('#m2').textContent=
        s.done+'/'+s.NT;
    $('#m3').textContent=s.coll;
    $('#m4').textContent=s.estop;
    $('#m5').textContent=s.msgs;
    $('#m6').textContent=
        s.fin||'-';
    $('#m7').textContent=s.restarts;
    $('#m8').textContent=s.slows;
    $('#m9').textContent=s.bund;
    $('#m10').textContent=s.jams;
    $('#m11').textContent=s.detours;
    $('#m13').textContent=s.dl;
    $('#m14').textContent=s.rer;
    $('#m15').textContent=s.dist;


    /* =====================================================
       EDGE AI DISPLAY
       ===================================================== */

    const k=risk(s);


    const aiColor=
        k.lvl=='LOW'
            ?'#22c55e'
            :k.lvl=='MEDIUM'
                ?'#f59e0b'
                :k.lvl=='HIGH'
                    ?'#ef4444'
                    :'#38bdf8';


    const probability=
        typeof k.probability==='number'
            ?(
                k.probability*100
            ).toFixed(0)
            :'-';


    $('#ai').innerHTML=`

<div style="
    font-size:22px;
    font-weight:700;
    color:${aiColor}
">
    ${k.lvl}
</div>


<div class="s" style="margin:0">

${
    k.best
        ?'Predicted close approach: R'+
         (k.best.a+1)+
         ' and R'+
         (k.best.b+1)+
         ' in '+
         k.best.t+
         ' s'
        :'No close approach predicted in the next 10 s'
}

</div>


<div class="s" style="margin:4px 0 0">

ML risk confidence:
${probability}%

· Local inference:
${k.ms.toFixed(2)} ms

</div>


<div class="s" style="margin:4px 0 0">

Model: Decision Tree ·
10 local features ·
No central server

</div>

`;
    

    /* =====================================================
       TASK ALLOCATION
       ===================================================== */

    $('#ta').innerHTML=
        s.tasks
            .filter(
                t=>t.st==1||t.st==2
            )
            .slice(0,8)
            .map(t=>{

                const r=
                    s.rs.find(
                        r=>
                            r.task==t||
                            r.t2==t
                    );


                return`
<div class="r">

<b>T${t.id+1}</b>

<span>
→ R${r?r.id+1:'?'}
</span>

<span style="
    flex:1;
    color:var(--mut)
">

${t.w} kg ·
${t.st==1
    ?'to pickup'
    :'carrying'}

</span>

<span>
urgency ${t.u}
</span>

</div>
`;
            })
            .join('')||
            '<span class="s">No active tasks</span>';


    /* =====================================================
       FLEET STATUS
       ===================================================== */

    $('#fl').innerHTML=
        '<table>'+
        '<tr>'+
        '<th>Robot</th>'+
        '<th>Task</th>'+
        '<th>Urgency</th>'+
        '<th>Position</th>'+
        '<th>Battery</th>'+
        '<th>Delay</th>'+
        '</tr>'+

        s.rs.map(r=>{

            const st=
                r.dead
                    ?'FAILED'
                    :r.bay
                        ?'charging'
                        :r.park
                            ?'parked'
                            :r.task
                                ?'T'+
                                 (r.task.id+1)+
                                 (
                                    r.t2
                                        ?'+T'+
                                         (r.t2.id+1)
                                        :''
                                 )+
                                 ' '+
                                 Math.round(
                                    r.task.w+
                                    (
                                        r.t2
                                            ?r.t2.w
                                            :0
                                    )
                                 )+
                                 'kg'+
                                 (
                                    r.slow
                                        ?' ⏬'
                                        :''
                                 )
                                :'idle';


            const u=
                r.task
                    ?Math.max(
                        r.task.u,
                        r.t2
                            ?r.t2.u
                            :0
                    )
                    :'-';


            return`
<tr>

<td>

<span
    class="dot"
    style="
        display:inline-block;
        vertical-align:middle;
        background:${COL[r.id]}
    "
></span>

R${r.id+1}

</td>

<td>${st}</td>

<td>${u}</td>

<td>
(${r.x},${r.y})
</td>

<td
    style="
        color:${
            r.bat<25
                ?'#ef4444'
                :'inherit'
        }
    "
>
${Math.round(r.bat)}%
</td>

<td>
${r.delay} s
</td>

</tr>
`;
        })
        .join('')+
        '</table>';


    cmpUI();


    $('#log').innerHTML=
        s.log.join('<br>');
}


/* =========================================================
   TIMER
   ========================================================= */

function tick(){

    clearInterval(iv);


    iv=setInterval(
        ()=>{
            if(run)
                sim.step();

            note();
            draw();
        },
        1000/+$('#sp').value
    );
}


/* =========================================================
   BUTTONS
   ========================================================= */

$('#pl').onclick=e=>{

    run=!run;

    e.target.textContent=
        run
            ?'Pause'
            :'Play';
};


$('#rs').onclick=()=>{

    cmp=0;

    seed++;

    reset();
};


$('#md').onchange=e=>{

    cmp=0;

    mode=e.target.value;

    reset();
};


$('#cm').onclick=()=>{

    cmp=1;

    mode='stop';

    $('#md').value='stop';

    run=true;

    $('#pl').textContent='Pause';

    reset();
};


$('#sp').onchange=tick;


$('#ls').onchange=()=>{

    sim.loss=
        +$('#ls').value/100;
};


$('#nr').onchange=()=>{

    cmp=0;

    reset();
};


$('#ac').onchange=()=>{

    ACC=
        +$('#ac').value;

    cmp=0;

    reset();
};


$('#cp').onchange=()=>{

    sim.cap=
        +$('#cp').value;
};


/* Block random aisle */

$('#bk').onclick=()=>{

    const l=
        PICKS.filter(
            c=>!sim.blocks.has(c)
        );


    sim.toggle(
        l[
            Math.random()*
            l.length|0
        ]
    );

    draw();
};


/* Traffic jam */

$('#jm').onclick=()=>{

    sim.mkjam();

    draw();
};


/* Fail robot */

$('#kl').onclick=()=>{

    sim.kill();

    draw();
};


/* Click map */

cv.onclick=e=>{

    if(view!='home')
        return;


    const b=
        cv.getBoundingClientRect();


    sim.toggle(
        idx(
            Math.floor(
                (e.clientX-b.left)/
                b.width*
                W
            ),
            Math.floor(
                (e.clientY-b.top)/
                b.height*
                H
            )
        )
    );


    draw();
};


/* =========================================================
   10 JOB SET BENCHMARK
   ========================================================= */

$('#bn').onclick=()=>{

    $('#bo').textContent=
        'Running 10 job sets…';


    setTimeout(()=>{

        const N=+$('#nr').value,
        cap=+$('#cp').value,
        G=[];

        let cl=0;

        let h=
            '<table>'+
            '<tr>'+
            '<th>Job set</th>'+
            '<th>HiveLane</th>'+
            '<th>Stop-and-wait</th>'+
            '<th>Faster</th>'+
            '<th>vs traditional</th>'+
            '<th>Coll.</th>'+
            '</tr>';


        for(let i=1;i<=10;i++){

            const o=
                bench(
                    i,
                    N,
                    cap
                );


            const g=
                (
                    1-
                    o.hive.fin/
                    o.stop.fin
                )*100;


            G.push(g);

            cl+=o.hive.coll;


            h+=`
<tr>

<td>${i}</td>

<td>
${o.hive.fin} s
</td>

<td>
${o.stop.fin} s
</td>

<td>
${g.toFixed(0)}%
</td>

<td>
${
    (
        1-
        o.hive.fin/
        o.trad.fin
    ).toFixed(0)
}%
</td>

<td>
${o.hive.coll}
</td>

</tr>
`;
        }


        $('#bo').innerHTML=
            h+
            `</table>

<p>

<b>
Average:
${
    (
        G.reduce(
            (a,b)=>a+b
        )/10
    ).toFixed(1)
}%
faster than stop-and-wait
</b>

(range
${Math.min(...G).toFixed(0)}
to
${Math.max(...G).toFixed(0)}%).

${G.filter(
    x=>x>=20
).length}
of 10 job sets reached 20%.

Collisions:
${cl}.

${N} robots,
${cap} kg payload.

"Traditional" =
stop-and-wait with one job per trip.

</p>`;

    },30);
};


/* =========================================================
   SCENARIO INFORMATION
   ========================================================= */

const INFO={

    dl:[
        'Deadlock scenario',

        '3 robots reach one junction at the same moment, all heading east. Stop-and-wait: they block each other and wait. HiveLane: robots share their plans 10 s ahead, so the urgent robot (R1) goes first at full speed, R3 slows down and follows, then R2. Nobody stops.'
    ],

    co:[
        'Collision scenario',

        '2 robots meet face to face in a single lane. Stop-and-wait: both stop and wait until a timeout. HiveLane: they knew 10 s ahead, the lower-priority robot slows down and uses a side gap, the other passes. No emergency stop needed.'
    ]
};


/* =========================================================
   SCENARIO NAVIGATION
   ========================================================= */

function go(v){

    view=v;

    cmp=0;

    document.body.classList.toggle(
        'sc',
        v!='home'
    );


    if(v!='home'){

        $('#st').textContent=
            INFO[v][0];

        $('#sdsc').textContent=
            INFO[v][1];
    }


    run=true;

    $('#pl').textContent='Pause';

    reset();

    window.scrollTo(
        0,
        0
    );
}


$('#gdl').onclick=()=>
    go('dl');


$('#gco').onclick=()=>
    go('co');


$('#bk2').onclick=()=>
    go('home');


/* =========================================================
   START
   ========================================================= */

reset();

tick();


/* =========================================================
   REFRESH UI AFTER EDGE-AI MODEL LOADS
   ========================================================= */

loadEdgeAIModel().then(model=>{

    if(model)
        draw();

});
}