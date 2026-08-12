import { Typography } from 'antd'

const { Title, Paragraph } = Typography

export default function PageHeading({ title, description, actions }) {
  return (
    <div className="page-heading">
      <div>
        <Title level={2}>{title}</Title>
        {description && <Paragraph>{description}</Paragraph>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}
