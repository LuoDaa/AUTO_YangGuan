import React, { Suspense, useMemo } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { Bounds, OrbitControls } from '@react-three/drei'
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
      <meshStandardMaterial color={color} roughness={0.72} metalness={0.02} transparent={opacity < 1} opacity={opacity} />
    </mesh>
  )
}

export default function Viewer({ assets, visible = {} }) {
  const meshes = [
    assets?.target && { key:'target', url:assets.target, color:'#d7dce2', opacity:0.72 },
    assets?.opponent && { key:'opponent', url:assets.opponent, color:'#6f7782', opacity:0.35 },
    assets?.support && { key:'support', url:assets.support, color:'#8f5d3b', opacity:1 },
    assets?.outer && { key:'outer', url:assets.outer, color:'#f2c94c', opacity:1 },
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
              {meshes.map(m => (
                <StlMesh key={m.key} {...m} visible={visible[m.key] !== false} />
              ))}
            </group>
          </Bounds>
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>
    </div>
  )
}
