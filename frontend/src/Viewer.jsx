import React, { Suspense, useMemo } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { Bounds, Line, OrbitControls, Sphere } from '@react-three/drei'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

function StlMesh({ url, color, opacity = 1, visible = true }) {
  const geometry = useLoader(STLLoader, url)
  useMemo(() => {
    geometry.computeVertexNormals()
    geometry.computeBoundingBox()
    geometry.computeBoundingSphere()
  }, [geometry])
  if (!visible) return null
  return (
    <mesh geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={0.72} metalness={0.02} transparent={opacity < 1} opacity={opacity} depthWrite={opacity > 0.55} />
    </mesh>
  )
}

function add(a, b, scale = 1) {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale]
}

function LocalizationOverlay({ localization, visible = {} }) {
  if (!localization) return null
  const frame = localization.frame || {}
  const origin = frame.origin
  const margin = localization.margin_points || []
  if (!origin || !frame.x || !frame.y || !frame.z) return null

  let axisLength = 7
  if (margin.length > 3) {
    const distances = margin.map(p => Math.hypot(p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]))
    axisLength = Math.max(4.5, Math.min(10, Math.max(...distances) * 1.8))
  }
  const closedMargin = margin.length > 2 ? [...margin, margin[0]] : []

  return <group>
    {visible.margin !== false && closedMargin.length > 2 && <Line points={closedMargin} color="#39d98a" lineWidth={3.2} />}
    {visible.frame !== false && <>
      <Sphere args={[0.16, 18, 18]} position={origin}><meshStandardMaterial color="#ffffff" /></Sphere>
      <Line points={[origin, add(origin, frame.x, axisLength)]} color="#ff6b6b" lineWidth={3} />
      <Line points={[origin, add(origin, frame.y, axisLength)]} color="#4dabf7" lineWidth={3} />
      <Line points={[origin, add(origin, frame.z, axisLength)]} color="#e599f7" lineWidth={4} />
    </>}
  </group>
}

export default function Viewer({ assets, visible = {}, localization = null }) {
  const meshes = [
    assets?.target && { key:'target', url:assets.target, color:'#d7dce2', opacity:0.66 },
    assets?.opponent && { key:'opponent', url:assets.opponent, color:'#6f7782', opacity:0.28 },
    assets?.support && { key:'support', url:assets.support, color:'#9a613f', opacity:1 },
    assets?.mesial && { key:'mesial', url:assets.mesial, color:'#40c057', opacity:0.8 },
    assets?.distal && { key:'distal', url:assets.distal, color:'#20c997', opacity:0.8 },
    assets?.placed && { key:'placed', url:assets.placed, color:'#b58df3', opacity:1 },
    assets?.height_fit && { key:'height_fit', url:assets.height_fit, color:'#74c0fc', opacity:1 },
    assets?.mesial_fit && { key:'mesial_fit', url:assets.mesial_fit, color:'#69db7c', opacity:1 },
    assets?.distal_fit && { key:'distal_fit', url:assets.distal_fit, color:'#ffa94d', opacity:1 },
    assets?.occlusal && { key:'occlusal', url:assets.occlusal, color:'#f783ac', opacity:1 },
    assets?.outer && { key:'outer', url:assets.outer, color:'#f2c94c', opacity:1 },
    assets?.blockout && { key:'blockout', url:assets.blockout, color:'#37bdf8', opacity:0.38 },
    assets?.cutter && { key:'cutter', url:assets.cutter, color:'#62d6c4', opacity:0.45 },
    assets?.intaglio_candidate && { key:'intaglio_candidate', url:assets.intaglio_candidate, color:'#e599f7', opacity:0.95 },
  ].filter(Boolean)

  return (
    <div className="viewer-shell">
      <Canvas shadows camera={{ position:[0, 0, 120], fov:35 }}>
        <color attach="background" args={['#11161d']} />
        <ambientLight intensity={1.4} />
        <directionalLight position={[50, 80, 100]} intensity={2.2} />
        <directionalLight position={[-50, -40, 30]} intensity={0.8} />
        <Suspense fallback={null}>
          <Bounds fit clip observe margin={1.15}>
            <group>
              {meshes.map(m => <StlMesh key={m.key} {...m} visible={visible[m.key] !== false} />)}
              <LocalizationOverlay localization={localization} visible={visible} />
            </group>
          </Bounds>
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>
    </div>
  )
}
