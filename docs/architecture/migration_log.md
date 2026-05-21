# Migration Log

## Fase 0: Blueprint

Estado: completada.

Entregables:

- `docs/architecture/implementation_blueprint.md`
- Definicion de topologia objetivo: Modular Monolith + Pipeline local
- Frontera nueva de codigo: `src/modulos`
- Mapa de migracion inicial
- Contratos de datos iniciales

## Fase 1: Reorganizacion Inicial

Estado: completada parcialmente.

Cambios aplicados:

- `src/01` se movio a `course_notes/01`.
- `src/02` se movio a `course_notes/02`.
- `docs/01/notes` se movio a `course_notes/01/notes`.
- `docs/01/work_files` se movio a `course_notes/01/work_files`.
- PDFs de `docs/01` se movieron a `course_notes/01/outputs`.
- `docs/02/notes` se movio a `course_notes/02/notes`.
- PDFs de `docs/02/outputs` se movieron a `course_notes/02/outputs`.
- `src/proyect` se movio a `course_notes/legacy_projects/option_hedging_original`.
- Se creo la estructura base para `projects/option_hedging`.
- Se creo la estructura base para `src/modulos`.
- Se actualizaron referencias basicas en `docs/basic_commands.md`.

Pendiente intencional:

- `src/cuantis_utils` se mantiene como legado temporal.
- `src/data` se mantiene temporalmente hasta la fase de storage.
- No se corrigieron imports ni rutas internas de notebooks.
- No se extrajo todavia la logica de ThetaData desde `clean_data.ipynb`.
- No se implementaron modelos, estrategias ni pipelines nuevos.

Riesgo conocido:

- Algunos notebooks legados contienen rutas hardcodeadas como `src/proyect`.
  Esas rutas se corregiran cuando se cree la version publicable en
  `projects/option_hedging`.
