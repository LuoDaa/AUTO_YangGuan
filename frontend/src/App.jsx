import React, { useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import Viewer from './Viewer.jsx'

const FLOW = [
  {id:'home', n:1, label:'首页'},
  {id:'case', n:2, label:'订单'},
  {id:'design', n:3, label:'自动设计 / 导出'},
]

function Topbar({ step, caseData, busy, goHome }) {
  return <header className="topbar">
    <button className="brand" onClick={goHome}><span className="logo">Y</span><b>AUTO_YangGuan</b></button>
    <div className="crumb">{caseData ? `${caseData.name} / FDI ${caseData.fdi}` : '请选择订单'}</div>
    <span className={`state ${busy ? 'busy' : ''}`}>{busy ? '处理中…' : '就绪'}</span>
  </header>
}

function Stepper({ step }) {
  return <div className="stepper">{FLOW.map(item => <div key={item.id} className={`step ${step === item.id ? 'active' : ''} ${FLOW.findIndex(x=>x.id===step) > FLOW.findIndex(x=>x.id===item.id) ? 'done' : ''}`}>
    <span>{item.n}</span><b>{item.label}</b>
  </div>)}</div>
}

function Home({ testCases, onPickTest, onUpload, busy }) {
  const [fdi, setFdi] = useState('36')
  const [upper, setUpper] = useState(null)
  const [lower, setLower] = useState(null)
  return <div className="page intake">
    <section className="hero">
      <div><small>AUTO CROWN MVP</small><h1>牙冠自动生成测试台</h1><p>流程按 CrownFlow 的基台快速订单：选择病例 → 牙冠生成 → 3D 检查 / 导出。</p></div>
      <div className="hero-badge">20–40 分保守候选</div>
    </section>

    <div className="grid two">
      <section className="panel">
        <div className="panel-head"><div><small>TEST DATA</small><h2>仓库测试病例</h2></div><span>{testCases.length} 例</span></div>
        <div className="case-list">
          {testCases.map(item => <button disabled={busy} key={item.name} className="case-card" onClick={() => onPickTest(item)}>
            <span className="fdi">{item.fdi}</span><div><b>{item.name}</b><small>{item.files.join(' · ')}</small></div><em>进入订单 →</em>
          </button>)}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><div><small>NEW ORDER</small><h2>上传本地病例</h2></div></div>
        <label className="field"><span>目标牙位 FDI</span><input value={fdi} onChange={e=>setFdi(e.target.value.replace(/\D/g,'').slice(0,2))} /></label>
        <label className="upload"><span>上颌 STL</span><input type="file" accept=".stl,.STL" onChange={e=>setUpper(e.target.files?.[0]||null)} /><b>{upper?.name || '选择文件'}</b></label>
        <label className="upload"><span>下颌 STL</span><input type="file" accept=".stl,.STL" onChange={e=>setLower(e.target.files?.[0]||null)} /><b>{lower?.name || '选择文件'}</b></label>
        <button className="primary full" disabled={busy || !upper || !lower || fdi.length !== 2} onClick={() => onUpload({fdi, upper, lower})}>创建订单</button>
      </section>
    </div>
  </div>
}

function CasePage({ caseData, onGenerate, busy, message }) {
  return <div className="page order-page">
    <div className="page-title"><div><small>NEW CROWN ORDER</small><h1>新建牙冠病例</h1><p>自动选择 FDI 对应模板；点击“牙冠生成”后一次完成模板裁冠、定位和对颌避让。</p></div><div className="order-meta"><span>FDI</span><b>{caseData.fdi}</b></div></div>
    <div className="grid order-grid">
      <section className="panel preview-panel">
        <div className="panel-head"><div><small>CASE PREVIEW</small><h2>上下颌病例</h2></div></div>
        <Viewer assets={caseData.assets} visible={{target:true, opponent:true}} />
      </section>
      <section className="panel controls-panel">
        <div className="panel-head"><div><small>TEMPLATE</small><h2>牙型与生成策略</h2></div></div>
        <div className="template-card"><div className="tooth-icon">{caseData.fdi}</div><div><b>FDI_{caseData.fdi}.stl</b><small>32 牙位基础解剖模板库 · 自动裁为 crown prior</small></div><span>已选择</span></div>
        <div className="strategy">
          <div><b>模板</b><span>轻微缩放，优先保形</span></div>
          <div><b>邻牙</b><span>当前 MVP 暂不做整颌自动分牙</span></div>
          <div><b>对颌</b><span>仅消除明显穿透</span></div>
          <div><b>内冠</b><span>本轮先测外冠，不参与外形反推</span></div>
        </div>
        <div className="notice">{message || '病例已就绪。点击下方“牙冠生成”开始。'}</div>
        <button className="primary generate" disabled={busy} onClick={onGenerate}>{busy ? '正在生成牙冠…' : '牙冠生成'}</button>
      </section>
    </div>
  </div>
}

function DesignPage({ caseData, busy, onBack }) {
  const [visible, setVisible] = useState({target:true, opponent:true, support:true, outer:true})
  const report = caseData.report || {}
  const qc = report.outer_qc || {}
  return <div className="page design-page">
    <div className="design-toolbar">
      <button className="secondary" onClick={onBack}>← 返回订单</button>
      <div><b>{caseData.name}</b><small>FDI {caseData.fdi} · {caseData.status}</small></div>
      <a className="primary export" href={caseData.assets.outer} download>导出 Outer STL</a>
    </div>
    <div className="design-layout">
      <section className="viewer-panel"><Viewer assets={caseData.assets} visible={visible} /></section>
      <aside className="side-panel">
        <div><small>DISPLAY</small><h3>显示</h3></div>
        {[
          ['target','目标颌'],['opponent','对颌'],['support','基台 component'],['outer','自动外冠']
        ].map(([key,label]) => <label className="toggle" key={key}><input type="checkbox" checked={visible[key]} onChange={e=>setVisible(v=>({...v,[key]:e.target.checked}))}/><span>{label}</span></label>)}
        <hr/>
        <div><small>QC</small><h3>工程检查</h3></div>
        <div className="metrics">
          <div><span>状态</span><b>{qc.status || caseData.status}</b></div>
          <div><span>最大位移</span><b>{qc.max_displacement_mm != null ? `${Number(qc.max_displacement_mm).toFixed(2)} mm` : '—'}</b></div>
          <div><span>P95 位移</span><b>{qc.p95_displacement_mm != null ? `${Number(qc.p95_displacement_mm).toFixed(2)} mm` : '—'}</b></div>
        </div>
        <div className="warn"><b>当前测试定位</b><p>这是 20–40 分外冠 MVP。Margin 为快速代理线，邻牙自动分牙和正式内冠尚未接入；先用来判断模板保形路线是否值得继续。</p></div>
        {caseData.assets.report && <a className="link" href={caseData.assets.report} target="_blank">打开 report.json</a>}
      </aside>
    </div>
  </div>
}

export default function App() {
  const [step, setStep] = useState('home')
  const [testCases, setTestCases] = useState([])
  const [caseData, setCaseData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => { api.testCases().then(x => setTestCases(x.cases || [])).catch(e => setMessage(e.message)) }, [])
  const title = useMemo(() => FLOW.find(x=>x.id===step)?.label || '', [step])

  async function pickTest(item) {
    setBusy(true); setMessage(`正在载入 ${item.name}…`)
    try { const c = await api.createFromTestdata(item.name); setCaseData(c); setStep('case'); setMessage('病例已载入，可直接生成。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  async function upload(payload) {
    setBusy(true); setMessage('正在上传病例…')
    try { const c = await api.uploadCase(payload); setCaseData(c); setStep('case'); setMessage('病例已载入，可直接生成。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  async function generate() {
    if (!caseData) return
    setBusy(true); setMessage('正在执行：基台识别 → 模板裁冠 → 自动摆放 → 对颌避让 → QC…')
    try { const c = await api.generate(caseData.id); setCaseData(c); setStep('design'); setMessage('外冠候选已生成。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  function home(){ setStep('home'); setCaseData(null); setMessage('') }

  return <div className="app">
    <Topbar step={step} caseData={caseData} busy={busy} goHome={home}/>
    {step !== 'home' && <Stepper step={step}/>} 
    <main>
      {step === 'home' && <Home testCases={testCases} onPickTest={pickTest} onUpload={upload} busy={busy}/>} 
      {step === 'case' && caseData && <CasePage caseData={caseData} onGenerate={generate} busy={busy} message={message}/>} 
      {step === 'design' && caseData && <DesignPage caseData={caseData} busy={busy} onBack={()=>setStep('case')}/>} 
    </main>
    {message && <div className="toast">{message}<button onClick={()=>setMessage('')}>×</button></div>}
    {busy && <div className="overlay"><div className="loader"></div><b>正在处理病例</b><span>{message}</span></div>}
    <footer className="footer"><span>AUTO_YangGuan · {title}</span><span>本地研发测试，不用于临床</span></footer>
  </div>
}
